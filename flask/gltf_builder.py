import numpy as np
import pygltflib

points = np.array(
    [
        [0, -1.5, 1.5],
        [3, -1.5, 1.5],
        [0, 1.5, 1.5],
        [3, 1.5, 1.5],
        [3, -1.5, -1.5],
        [0, -1.5, -1.5],
        [3, 1.5, -1.5],
        [0, 1.5, -1.5],
    ],
    dtype="float32",
)
triangles = np.array(
    [
        [0, 1, 2],
        [3, 2, 1],
        [1, 0, 4],
        [5, 4, 0],
        [3, 1, 6],
        [4, 6, 1],
        [2, 3, 7],
        [6, 7, 3],
        [0, 2, 5],
        [7, 5, 2],
        [5, 7, 4],
        [6, 4, 7],
    ],
    dtype="uint8",
)

def mesh_to_glb(points, triangles):
	triangles_binary_blob = triangles.flatten().tobytes()
	points_binary_blob = points.tobytes()
	gltf = pygltflib.GLTF2(
		scene=0,
		scenes=[pygltflib.Scene(nodes=[0])],
		nodes=[pygltflib.Node(mesh=0)],
		meshes=[
			pygltflib.Mesh(
				primitives=[
					pygltflib.Primitive(
						attributes=pygltflib.Attributes(POSITION=1), indices=0
					)
				]
			)
		],
		accessors=[
			pygltflib.Accessor(
				bufferView=0,
				componentType=pygltflib.UNSIGNED_BYTE,
				count=triangles.size,
				type=pygltflib.SCALAR,
				max=[int(triangles.max())],
				min=[int(triangles.min())],
			),
			pygltflib.Accessor(
				bufferView=1,
				componentType=pygltflib.FLOAT,
				count=len(points),
				type=pygltflib.VEC3,
				max=points.max(axis=0).tolist(),
				min=points.min(axis=0).tolist(),
			),
		],
		bufferViews=[
			pygltflib.BufferView(
				buffer=0,
				byteLength=len(triangles_binary_blob),
				target=pygltflib.ELEMENT_ARRAY_BUFFER,
			),
			pygltflib.BufferView(
				buffer=0,
				byteOffset=len(triangles_binary_blob),
				byteLength=len(points_binary_blob),
				target=pygltflib.ARRAY_BUFFER,
			),
		],
		buffers=[
			pygltflib.Buffer(
				byteLength=len(triangles_binary_blob) + len(points_binary_blob)
			)
		],
	)
	gltf.set_binary_blob(triangles_binary_blob + points_binary_blob)
	filename = "models/mesh.glb"
	filename2 = "models/mesh.gltf"
	gltf.save(filename)
	gltf.save(filename2)
	return filename

# mesh_to_glb(points, triangles)