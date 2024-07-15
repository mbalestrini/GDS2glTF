#!/usr/bin/env python3
"""
This program converts a GDSII 2D layout file to a glTF 3D file

USAGE:
    - edit the "layerstack" variable in the "CONFIGURATION" section below
    - run "gdsiigtlf file.gds"
OUTPUT:
    - the files file.gds.gltf

The program takes one argument, a path to a GDSII file. It reads shapes from
each layer of the GDSII file, converts them to polygon boundaries, then makes
a triangle mesh for each GDSII layer by extruding the polygons to given sizes.

All units, including the units of the exported file, are the GDSII file's
user units (often microns).
"""
import argparse # cmd-line parsing helper
import json # for loading layerstack from file
import sys # read command-line arguments
import gdspy # open gds file
import numpy as np # fast math on lots of points
import triangle # triangulate polygons



import pygltflib
from pygltflib import BufferFormat
from pygltflib.validator import validate, summary

def load_layerstack_from_file(layerstack_filepath):
    with open(layerstack_filepath) as lsf:
        json_data = json.load(lsf)
    layerstack = {}
    for entry in json_data:
        layer_type = (entry['layer'], entry['type'])
        entry['zmax'] = round(entry['zmin'] + entry['thickness'], 5)
        # entry.pop('thickness')              # used for development to ensure the sky130.layerstack file had same results as the hard-coded layerstack, for diffing
        # entry.pop('layer')                  # used for development to ensure the sky130.layerstack file had same results as the hard-coded layerstack, for diffing
        # entry.pop('type')                   # used for development to ensure the sky130.layerstack file had same results as the hard-coded layerstack, for diffing
        # entry['color'] = entry.pop('color') # used for development to ensure the sky130.layerstack file had same results as the hard-coded layerstack, for diffing
        layerstack[layer_type] = entry
    return layerstack

def do_main(gdsii_file_path, layerstack_file_path = None):
    ########## CONFIGURATION (EDIT THIS PART) #####################################

    # choose which GDSII layers to use
    if layerstack_file_path:
        layerstack = load_layerstack_from_file(layerstack_file_path)
    else:
        layerstack = {    
            (235,4): {'name':'substrate', 'zmin':-2, 'zmax':0, 'color':[ 0.2, 0.2, 0.2, 1.0]},
            (64,20): {'name':'nwell', 'zmin':-0.5, 'zmax':0.01, 'color':[ 0.4, 0.4, 0.4, 1.0]},
            # (65,44): {'name':'tap', 'zmin':0, 'zmax':0.1, 'color':[ 0.4, 0.4, 0.4, 1.0]},
            (65,20): {'name':'diff', 'zmin':-0.12, 'zmax':0.02, 'color':[ 0.9, 0.9, 0.9, 1.0]},
            (66,20): {'name':'poly', 'zmin':0, 'zmax':0.18, 'color':[ 0.75, 0.35, 0.46, 1.0]},
            (66,44): {'name':'licon', 'zmin':0, 'zmax':0.936, 'color':[ 0.2, 0.2, 0.2, 1.0]},
            (67,20): {'name':'li1', 'zmin':0.936, 'zmax':1.136, 'color':[ 1.0, 0.81, 0.55, 1.0]},
            (67,44): {'name':'mcon', 'zmin':1.011, 'zmax':1.376, 'color':[ 0.2, 0.2, 0.2, 1.0]},
            (68,20): {'name':'met1', 'zmin':1.376, 'zmax':1.736, 'color':[ 0.16, 0.38, 0.83, 1.0]},
            (68,44): {'name':'via', 'zmin':1.736,'zmax':2, 'color':[ 0.2, 0.2, 0.2, 1.0]},
            (69,20): {'name':'met2', 'zmin':2, 'zmax':2.36, 'color':[ 0.65, 0.75, 0.9, 1.0]},
            (69,44): {'name':'via2', 'zmin':2.36, 'zmax':2.786, 'color':[ 0.2, 0.2, 0.2, 1.0]},
            (70,20): {'name':'met3', 'zmin':2.786, 'zmax':3.631, 'color':[ 0.2, 0.62, 0.86, 1.0]},
            (70,44): {'name':'via3', 'zmin':3.631, 'zmax':4.0211, 'color':[ 0.2, 0.2, 0.2, 1.0]},
            (71,20): {'name':'met4', 'zmin':4.0211, 'zmax':4.8661, 'color':[ 0.15, 0.11, 0.38, 1.0]},
            (71,44): {'name':'via4', 'zmin':4.8661, 'zmax':5.371, 'color':[ 0.2, 0.2, 0.2, 1.0]},
            (72,20): {'name':'met5', 'zmin':5.371, 'zmax':6.6311, 'color':[ 0.4, 0.4, 0.4, 1.0]}
            # (83,44): { 'zmin':0, 'zmax':0.1, 'name':'text'}
        }

    # layerstack = {    
    #     (235,4): {'name':'substrate', 'zmin':-1, 'zmax':0, 'color':[ 0.2, 0.2, 0.2, 1.0]},
    #     # (64,20): {'name':'nwell', 'zmin':0, 'zmax':0.1, 'color':[ 0.4, 0.4, 0.4, 1.0]},    
    #     # (65,44): {'name':'tap', 'zmin':0, 'zmax':0.1, 'color':[ 0.4, 0.4, 0.4, 1.0]},    
    #     (65,20): {'name':'diff', 'zmin':-0.12, 'zmax':0.01, 'color':[ 0.9, 0.9, 0.9, 1.0]},    
    #     (66,20): {'name':'poly', 'zmin':0, 'zmax':0.18, 'color':[ 0.4, 0.4, 0.4, 1.0]},    
    #     (66,44): {'name':'licon', 'zmin':0, 'zmax':0.936, 'color':[ 0.4, 0.4, 0.4, 1.0]},    
    #     (67,20): {'name':'li1', 'zmin':0.936, 'zmax':1.136, 'color':[ 0.9, 0.9, 0.9, 1.0]},    
    #     (67,44): {'name':'mcon', 'zmin':1.011, 'zmax':1.376, 'color':[ 0.2, 0.2, 0.2, 1.0]},    
    #     (68,20): {'name':'met1', 'zmin':1.376, 'zmax':1.736, 'color':[ 0.4, 0.4, 0.4, 1.0]},    
    #     (68,44): {'name':'via', 'zmin':1.736,'zmax':2, 'color':[ 0.2, 0.2, 0.2, 1.0]},    
    #     (69,20): {'name':'met2', 'zmin':2, 'zmax':2.36, 'color':[ 0.9, 0.9, 0.9, 1.0]},    
    #     (69,44): {'name':'via2', 'zmin':2.36, 'zmax':2.786, 'color':[ 0.2, 0.2, 0.2, 1.0]},    
    #     (70,20): {'name':'met3', 'zmin':2.786, 'zmax':3.631, 'color':[ 0.4, 0.4, 0.4, 1.0]},    
    #     (70,44): {'name':'via3', 'zmin':3.631, 'zmax':4.0211, 'color':[ 0.2, 0.2, 0.2, 1.0]},    
    #     (71,20): {'name':'met4', 'zmin':4.0211, 'zmax':4.8661, 'color':[ 0.9, 0.9, 0.9, 1.0]},    
    #     (71,44): {'name':'via4', 'zmin':4.8661, 'zmax':5.371, 'color':[ 0.2, 0.2, 0.2, 1.0]},    
    #     (72,20): {'name':'met5', 'zmin':5.371, 'zmax':6.6311, 'color':[ 0.4, 0.4, 0.4, 1.0]},
    #     # (83,44): { 'zmin':0, 'zmax':0.1, 'name':'text'},
    # }







    ########## INPUT ##############################################################

    # First, the input file is read using the gdspy library, which interprets the
    # GDSII file and formats the data Python-style.
    # See https://gdspy.readthedocs.io/en/stable/index.html for documentation.
    # Second, the boundaries of each shape (polygon or path) are extracted for
    # further processing.
    
    print('Reading GDSII file {}...'.format(gdsii_file_path))
    gdsii = gdspy.GdsLibrary()
    gdsii.read_gds(gdsii_file_path, units='import')


    gltf = pygltflib.GLTF2()
    scene = pygltflib.Scene()
    gltf.scenes.append(scene)
    buffer = pygltflib.Buffer()
    gltf.buffers.append(buffer)

    for layer in layerstack:
        mainMaterial = pygltflib.Material()
        mainMaterial.doubleSided = False
        mainMaterial.name = layerstack[layer]['name']
        mainMaterial.pbrMetallicRoughness =  {
                        "baseColorFactor": layerstack[layer]['color'],
                        "metallicFactor": 0.5,
                        "roughnessFactor": 0.5
                    }
        gltf.materials.append(mainMaterial)

    binaryBlob = bytes()

    print('Extracting polygons...')

    meshes_lib = {}

    for cell in gdsii.cells.values(): # loop through cells to read paths and polygons
        layers = {} # array to hold all geometry, sorted into layers

        print ("\nProcessing cell: ", cell.name)
        
        # $$$CONTEXT_INFO$$$ is a separate, non-standard compliant cell added
        # optionally by KLayout to store extra information not needed here.
        # see https://www.klayout.de/forum/discussion/1026/very-
        # important-gds-exported-from-k-layout-not-working-on-cadence-at-foundry
        if cell.name == '$$$CONTEXT_INFO$$$':
            continue # skip this cell

        print ("\tpaths loop. total paths:" , len(cell.paths))
        # loop through paths in cell
        for path in cell.paths:
            lnum = (path.layers[0],path.datatypes[0]) # GDSII layer number
            
            if not lnum in layerstack.keys():
                continue

            layers[lnum] = [] if not lnum in layers else layers[lnum]
            # add paths (converted to polygons) that layer
            for poly in path.get_polygons():
                layers[lnum].append((poly, None, False))

        print ("\tpolygons loop. total polygons:" , len(cell.polygons))
        # loop through polygons (and boxes) in cell
        for polygon in cell.polygons:
            lnum = (polygon.layers[0],polygon.datatypes[0]) # same as before...

            if not lnum in layerstack.keys():
                continue

            layers[lnum] = [] if not lnum in layers else layers[lnum]
            for poly in polygon.polygons:
                layers[lnum].append((poly, None, False))


        
        
        """
        At this point, "layers" is a Python dictionary structured as follows:

        layers = {
        0 : [ ([[x1, y1], [x2, y2], ...], None, False), ... ]
        1 : [ ... ]
        2 : [ ... ]
        ...
        }

        Each dictionary key is a GDSII layer number (0-255), and the value of the
        dictionary at that key (if it exists; keys were only created for layers with
        geometry) is a list of polygons in that GDSII layer. Each polygon is a 3-tuple
        whose first element is a list of points (2-element lists with x and y
        coordinates), second element is None (for the moment; this will be used later),
        and third element is False (whether the polygon is clockwise; will be updated).
        """

        ########## TRIANGULATION ######################################################

        # An STL file is a list of triangles, so the polygons need to be filled with
        # triangles. This is a surprisingly hard algorithmic problem, especially since
        # there are few limits on what shapes GDSII file polygons can be. So we use the
        # Python triangle library (documentation is at https://rufat.be/triangle/),
        # which is a Python interface to a fast and well-written C library also called
        # triangle (with documentation at https://www.cs.cmu.edu/~quake/triangle.html).

        print('\tTriangulating polygons...')

        
        num_triangles = {} # will store the number of triangles for each layer

        # loop through all layers
        for layer_number, polygons in layers.items():

            # but skip layer if it won't be exported
            if not layer_number in layerstack.keys():
                continue

            num_triangles[layer_number] = 0

            # loop through polygons in layer
            for index, (polygon, _, _) in enumerate(polygons):

                num_polygon_points = len(polygon)

                # determine whether polygon points are CW or CCW
                area = 0
                for i, v1 in enumerate(polygon): # loop through vertices
                    v2 = polygon[(i+1) % num_polygon_points]
                    area += (v2[0]-v1[0])*(v2[1]+v1[1]) # integrate area

                clockwise = area > 0

                # GDSII implements holes in polygons by making the polygon edge
                # wrap into the hole and back out along the same line. However,
                # this confuses the triangulation library, which fills the holes
                # with extra triangles. Avoid this by moving each edge back a
                # very small amount so that no two edges of the same polygon overlap.
                delta = 0.00001 # inset each vertex by this much (smaller has broken one file)
                points_i = polygon # get list of points
                points_j = np.roll(points_i, -1, axis=0) # shift by 1
                points_k = np.roll(points_i, 1, axis=0) # shift by -1
                # calculate normals for each edge of each vertex (in parallel, for speed)
                normal_ij = np.stack((points_j[:, 1]-points_i[:, 1],
                                    points_i[:, 0]-points_j[:, 0]), axis=1)
                normal_ik = np.stack((points_i[:, 1]-points_k[:, 1],
                                    points_k[:, 0]-points_i[:, 0]), axis=1)
                length_ij = np.linalg.norm(normal_ij, axis=1)
                length_ik = np.linalg.norm(normal_ik, axis=1)
                normal_ij /= np.stack((length_ij, length_ij), axis=1)
                normal_ik /= np.stack((length_ik, length_ik), axis=1)
                if clockwise:
                    normal_ij = -1*normal_ij
                    normal_ik = -1*normal_ik
                # move each vertex inward along its two edge normals
                polygon = points_i - delta*normal_ij - delta*normal_ik

                # In an extreme case of the above, the polygon edge doubles back on
                # itself on the same line, resulting in a zero-width segment. I've
                # seen this happen, e.g., with a capital "N"-shaped hole, where
                # the hole split line cuts out the "N" shape but splits apart to
                # form the triangle cutout in one side of the shape. In any case,
                # simply moving the polygon edges isn't enough to deal with this;
                # we'll additionally mark points just outside of each edge, between
                # the original edge and the delta-shifted edge, as outside the polygon.
                # These parts will be removed from the triangulation, and this solves
                # just this case with no adverse affects elsewhere.
                hole_delta = 0.00001 # small fraction of delta
                holes = 0.5*(points_j+points_i) - hole_delta*delta*normal_ij
                # HOWEVER: sometimes this causes a segmentation fault in the triangle
                # library. I've observed this as a result of certain various polygons.
                # Frustratingly, the fault can be bypassed by *rotating the polygons*
                # by like 30 degrees (exact angle seems to depend on delta values) or
                # moving one specific edge outward a bit. I have absolutely no idea
                # what is wrong. In the interest of stability over full functionality,
                # this is disabled. TODO: figure out why this happens and fix it.
                use_holes = False

                # triangulate: compute triangles to fill polygon
                point_array = np.arange(num_polygon_points)
                edges = np.transpose(np.stack((point_array, np.roll(point_array, 1))))
                if use_holes:
                    triangles = triangle.triangulate(dict(vertices=polygon,
                                                        segments=edges,
                                                        holes=holes), opts='p')
                else:
                    triangles = triangle.triangulate(dict(vertices=polygon,
                                                        segments=edges), opts='p')

                if not 'triangles' in triangles.keys():
                    triangles['triangles'] = []

                # each line segment will make two triangles (for a rectangle), and the polygon
                # triangulation will be copied on the top and bottom of the layer.
                num_triangles[layer_number] += num_polygon_points*2 + \
                                            len(triangles['triangles'])*2
                polygons[index] = (polygon, triangles, clockwise)



            # glTF Mesh creation

            zmin = layerstack[layer_number]['zmin']
            zmax = layerstack[layer_number]['zmax']
            layername = layerstack[layer_number]['name']
            node_name = cell.name + "_" + layername

            gltf_positions = []
            gltf_indices = []        
            indices_offset = 0
            for i,(_, poly_data, clockwise) in enumerate(polygons):         

                
                p_positions_top = np.insert(poly_data['vertices'], 2, zmax, axis=1)
                p_positions_bottom = np.insert( poly_data['vertices'] , 2, zmin, axis=1)
                
                p_positions = np.concatenate( (p_positions_top, p_positions_bottom) )
                p_indices_top = poly_data['triangles']
                p_indices_bottom = np.flip ((p_indices_top+len(p_positions_top)), axis=1 )
                
                ind_list_top = np.arange(len(p_positions_top))
                ind_list_bottom = np.arange(len(p_positions_top)) + len(p_positions_top)

                if(clockwise):
                    ind_list_top = np.flip(ind_list_top, axis=0)
                    ind_list_bottom = np.flip(ind_list_bottom, axis=0)

                p_indices_right = np.stack( (ind_list_bottom, np.roll(ind_list_bottom, -1, axis=0) , np.roll(ind_list_top, -1, axis=0)), axis=1 )
                p_indices_left = np.stack( ( np.roll(ind_list_top, -1, axis=0), ind_list_top , ind_list_bottom ) , axis=1)
                    
                p_indices = np.concatenate( (p_indices_top, p_indices_bottom, p_indices_right, p_indices_left) )

                if(len(gltf_positions)==0):
                    gltf_positions = p_positions
                else:
                    gltf_positions = np.append(gltf_positions , p_positions, axis=0)
                if(len(gltf_indices)==0):
                    gltf_indices = p_indices
                else:    
                    gltf_indices = np.append(gltf_indices, p_indices + indices_offset, axis=0)            
                indices_offset += len(p_positions)
        

            
            indices_binary_blob = gltf_indices.astype(np.uint32).flatten().tobytes() #triangles.flatten().tobytes()
            positions_binary_blob = gltf_positions.astype(np.float32).tobytes() #points.tobytes()

            bufferView1 = pygltflib.BufferView()
            bufferView1.buffer = 0
            bufferView1.byteOffset = len(binaryBlob)
            bufferView1.byteLength = len(indices_binary_blob)
            bufferView1.target = pygltflib.ELEMENT_ARRAY_BUFFER
            gltf.bufferViews.append(bufferView1)

            accessor1 = pygltflib.Accessor()
            accessor1.bufferView = len(gltf.bufferViews)-1
            accessor1.byteOffset = 0
            accessor1.componentType = pygltflib.UNSIGNED_INT
            accessor1.type = pygltflib.SCALAR
            accessor1.count = gltf_indices.size
            accessor1.max = [int(gltf_indices.max())]
            accessor1.min = [int(gltf_indices.min())]
            gltf.accessors.append(accessor1)

            binaryBlob = binaryBlob + indices_binary_blob

            bufferView2 = pygltflib.BufferView()
            bufferView2.buffer = 0
            bufferView2.byteOffset = len(binaryBlob)
            bufferView2.byteLength = len(positions_binary_blob)
            bufferView2.target = pygltflib.ARRAY_BUFFER
            gltf.bufferViews.append(bufferView2)

            positions_count = len(gltf_positions)
            accessor2 = pygltflib.Accessor()
            accessor2.bufferView =  len(gltf.bufferViews)-1
            accessor2.byteOffset = 0
            accessor2.componentType = pygltflib.FLOAT
            accessor2.count = positions_count
            accessor2.type = pygltflib.VEC3
            accessor2.max = gltf_positions.max(axis=0).tolist()
            accessor2.min = gltf_positions.min(axis=0).tolist()
            gltf.accessors.append(accessor2)

            binaryBlob = binaryBlob + positions_binary_blob

            mesh = pygltflib.Mesh()
            mesh_primitive = pygltflib.Primitive()
            mesh_primitive.indices = len(gltf.accessors)-2
            mesh_primitive.attributes.POSITION = len(gltf.accessors)-1

            mesh_primitive.material = list(layerstack).index(layer_number)

            mesh.primitives.append(mesh_primitive)


            gltf.meshes.append(mesh)
            meshes_lib[node_name] = len(gltf.meshes)-1

        

        
    gltf.set_binary_blob(binaryBlob)
    buffer.byteLength = len(binaryBlob)
    gltf.convert_buffers(BufferFormat.DATAURI)





    def add_cell_node(c, parent_node, prefix):
        for ref in c.references:
            instance_node = pygltflib.Node()
            instance_node.extras = {}
            instance_node.extras["type"] = ref.ref_cell.name;
            if(ref.properties.get(61)==None):
                # ref.ref_cell.name
                instance_node.name = "???"; 
            else:
                instance_node.name = ref.properties[61]
                
            print(prefix, instance_node.name, "(", ref.ref_cell.name + ")")
            instance_node.translation = [ref.origin[0], ref.origin[1], 0]
            if(ref.rotation!=None):
                if(ref.rotation==90):
                    instance_node.rotation = [ 0, 0, 0.7071068, 0.7071068 ]
                elif(ref.rotation==180):
                    instance_node.rotation = [ 0, 0, 1, 0 ]
                elif(ref.rotation==270):
                    instance_node.rotation = [ 0, 0, 0.7071068, -0.7071068 ]
            if(ref.x_reflection):
                instance_node.scale = [1,-1,1]

            for layer in layerstack.values():
                lib_name = ref.ref_cell.name + "_" + layer['name']
                if(meshes_lib.get(lib_name)!=None):
                    layer_node = pygltflib.Node()
                    layer_node.name = lib_name
                    layer_node.mesh = meshes_lib[lib_name]
                    gltf.nodes.append(layer_node)
                    instance_node.children.append(len(gltf.nodes)-1)
            
            if(len(ref.ref_cell.references)>0):
                add_cell_node(ref.ref_cell, instance_node, prefix + "\t")

            gltf.nodes.append(instance_node)
            parent_node.children.append(len(gltf.nodes)-1)


    main_cell = gdsii.top_level()[0]

    root_node = pygltflib.Node()
    root_node.name = main_cell.name #"ROOT"
    gltf.nodes.append(root_node)

    print ("\nBuilding Scenegraph:")
    print(root_node.name)

    add_cell_node(main_cell, root_node, "\t")
        

    for layer in layerstack.values():
        lib_name = main_cell.name + "_" + layer['name']
        if(meshes_lib.get(lib_name)!=None):
            layer_node = pygltflib.Node()
            layer_node.name =lib_name
            layer_node.mesh = meshes_lib[lib_name]
            gltf.nodes.append(layer_node)
            root_node.children.append(len(gltf.nodes)-1)




    scene.nodes.append(0)
    gltf.scene = 0

    # validate(gltf)  # will throw an error depending on the problem
    # summary(gltf) 


    print ("\nWriting glTF file:")
    gltf.save(gdsii_file_path + ".gltf")
    # gltf.save("output.gltf")

    print('Done.')


if __name__ == '__main__':
    # get the input file name
    parser = argparse.ArgumentParser(description='Translate GDSII to WebGL for visualization.')
    parser.add_argument('gdsii_file_path', help='Input GDSII file.', type=str)
    parser.add_argument('-l', '--layerstack', required=False, help='Optional path to a "layerstack" file.')
    args = parser.parse_args()

    
    do_main(args.gdsii_file_path, args.layerstack)
