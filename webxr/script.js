import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

function onSecondFirstFrame(time, delta) {
	if (Math.floor(time/1000) != Math.floor((time - delta)/1000)) {
		return Math.floor(time/1000);
	}
}

function flipImageVertically(data, width, height) {
	var halfHeight = height / 2 | 0;  // the | 0 keeps the result an int
	var bytesPerRow = width * 4;

	// make a temp buffer to hold one row
	var temp = new Uint8Array(width * 4);
	for (var y = 0; y < halfHeight; ++y) {
		var topOffset = y * bytesPerRow;
		var bottomOffset = (height - y - 1) * bytesPerRow;

		// make copy of a row on the top half
		temp.set(data.subarray(topOffset, topOffset + bytesPerRow));

		// copy a row from the bottom half to the top
		data.copyWithin(topOffset, bottomOffset, bottomOffset + bytesPerRow);

		// copy the copy of the top half row to the bottom half 
		data.set(temp, bottomOffset);
	}
	return data;
}

async function createImageFromTexture(gl, texture, width, height) {
	// Create a framebuffer backed by the texture
	var framebuffer = gl.createFramebuffer();
	gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
	gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);

	// Read the contents of the framebuffer
	var data = new Uint8Array(width * height * 4);
	gl.readPixels(0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, data);

	gl.deleteFramebuffer(framebuffer);
	
	// ali: reverse
	data = flipImageVertically(data, width, height);

	// Create a 2D canvas to store the result 
	var canvas = document.createElement('canvas');
	canvas.width = width;
	canvas.height = height;
	var context = canvas.getContext('2d');

	// Copy the pixels to a 2D canvas
	var imageData = context.createImageData(width, height);
	imageData.data.set(data);
	context.putImageData(imageData, 0, 0);

	// Turn into Blob
	const imageBlob = await new Promise((resolve) =>
		canvas.toBlob(resolve, 'image/jpeg', 0.75)
	);

	return imageBlob;
}

function getCameraIntrinsics(projectionMatrix, viewport) {
	const p = projectionMatrix.elements;
  
	// Principal point in pixels (typically at or near the center of the viewport)
	let u0 = (1 - p[8]) * viewport.width / 2 + viewport.x;
	let v0 = (1 - p[9]) * viewport.height / 2 + viewport.y;
  
	// Focal lengths in pixels (these are equal for square pixels)
	let ax = viewport.width / 2 * p[0];
	let ay = viewport.height / 2 * p[5];
  
	// Skew factor in pixels (nonzero for rhomboid pixels)
	let gamma = viewport.width / 2 * p[4];
  
	// Merge the calculated intrinsics
	const K = [	[ax,  gamma,  u0,  0],
				[0,   ay,     v0,  0],
				[0,   0,      1,   0]];
	return K;
}

async function getPointCloudFromImage(imageBlob, camera, viewport) {
	imageBlob = await imageBlob;
	const file = new File([imageBlob], 'capture.jpg', {type: imageBlob.type});
	const camPose = camera.getWorldPosition(new THREE.Vector3())
	const unproject = camera.matrixWorld.clone().multiply(camera.projectionMatrixInverse);
	const K = getCameraIntrinsics(camera.projectionMatrix, viewport);

	const formData  = new FormData();
	formData.append("captureCount", JSON.stringify(captureCount++));
	formData.append("image", file);
	formData.append("camPose", JSON.stringify(camPose.toArray()));
	formData.append("unprojectMatrix", JSON.stringify(unproject.clone().transpose().toArray()));
	formData.append("camMatrixWorld", JSON.stringify(camera.matrixWorld.clone().transpose().toArray()));
	formData.append("camProjectionMatrix", JSON.stringify(camera.projectionMatrix.clone().transpose().toArray()));
	formData.append("K", JSON.stringify(K));
	
	const hostname = window.location.hostname;
	const points = await fetch("https://" + hostname + ":5000/upload_image", {method:"POST", body:formData})
	.then(response => response.arrayBuffer())
	.then(buffer => {
		const decodedData = msgpack.decode(new Uint8Array(buffer));
		return decodedData;
	}).catch(err => {
		alert(err);
		const depthButton = document.getElementById('depth-button');
		depthButton.innerHTML = "Add Depth";
		depthButton.disabled = false;
	});

	return await points;
	return Array(20).fill().map(() => Array(20).fill(1));
}

function toScreen(ndc, vp) {
	const x = vp.width * (ndc.x + 1)/2 + vp.x;
	const y = vp.height * (ndc.y + 1)/2 + vp.y;
	const z = 1;
	return new THREE.Vector3(x, y, z);
}

function toNDC(p_camera, vp) {
	const ndc_x = (p_camera.x - vp.x)/vp.width * 2 - 1;
	const ndc_y = (p_camera.y - vp.y)/vp.height * 2 - 1;
	const ndc_z = 1;
	return new THREE.Vector3(ndc_x, ndc_y, ndc_z);
}

async function addSmallCubesAt(points, scene, color=0x00FF00, opacity=0.2) {
	points = await points;
	// console.log(points);
	var dotGeometry = new THREE.BufferGeometry().setFromPoints(points);
	
	// const grayscale = Math.min(1, vector.length() / 5);
	// const dotColor = new THREE.Color(0, grayscale, 0);

	var voidMaterial = new THREE.PointsMaterial( { size: 0.05, color: 0x000000 } );
	voidMaterial.opacity = 0;
	voidMaterial.blending = THREE.MultiplyBlending;

	var voidDot = new THREE.Points( dotGeometry, voidMaterial );
	// voidDot.position.copy(scene.worldToLocal(vector));
	voidDot.name = "point";
	scene.add( voidDot );

	var colorMaterial = new THREE.PointsMaterial( { size: 0.05, color: color } );
	colorMaterial.transparent = true;
	colorMaterial.opacity = opacity;

	var colorDot = new THREE.Points( dotGeometry, colorMaterial );
	// colorDot.position.copy(scene.worldToLocal(vector));
	colorDot.name = "point";
	scene.add( colorDot );
}

function reconstructFromScreen(pixel, camera, viewport, depth) {
	const ndc = toNDC(pixel, viewport);
	const aspectRatioRoot = Math.sqrt(viewport.width / viewport.height);
	ndc.multiply(new THREE.Vector3(aspectRatioRoot, 1/aspectRatioRoot, 1));
	const rec = ndc.clone().unproject(camera);
	const camPose = camera.getWorldPosition(new THREE.Vector3());
	// rec = camera.localToWorld(camera.worldToLocal(rec).normalize().multiplyScalar(depth));
	rec.sub(camPose).normalize().multiplyScalar(depth).add(camPose);
	return rec;
}

async function addPointCloud(depth, scene, camera, viewport) {
	depth = await depth;

	const width = viewport.width;
	const height = viewport.height;
	const points = [];
	for (let i = 0; i < depth.length; i++) {
		for (let j = 0; j < depth[0].length; j++) {
			const y = height - Math.floor(height*(i+1)/depth.length);
			const x = Math.floor(width*j/depth[0].length);
			const pixel = new THREE.Vector3(x, y, 1);
			const rec = reconstructFromScreen(pixel, camera, viewport, depth[i][j]);
			points.push(scene.worldToLocal(rec));
		}
	}

	addSmallCubesAt(points, scene);
}

async function addDepthPointCloud(gl, session, referenceSpace, scene, camera, frame) {
	const depthButton = document.getElementById('depth-button');
	depthButton.innerHTML = "Waiting...";
	depthButton.disabled = true;

	camera = camera.clone();
	
	const pose = frame.getViewerPose(referenceSpace);
	const view = pose.views[0];
	const viewport = session.renderState.baseLayer.getViewport(view);

	const binding = new XRWebGLBinding(session, gl);
	const cameraTexture = binding.getCameraImage(view.camera);
	const imageBlob = createImageFromTexture(gl, cameraTexture, view.camera.width, view.camera.height);
	
	const points = await getPointCloudFromImage(imageBlob, camera, viewport);

	scene.getObjectsByProperty("name", "point").forEach((x) => scene.remove(x));
	// addPointCloud(depth, scene, camera, viewport);
	const vectorPoints = [];
	points[0].forEach((P) => vectorPoints.push(new THREE.Vector3(P[0], P[1], P[2])));
	addSmallCubesAt(vectorPoints, scene);

	const vectorKeyPoints = []
	points[1].forEach((P) => vectorKeyPoints.push(new THREE.Vector3(P[0], P[1], P[2])));
	addSmallCubesAt(vectorKeyPoints, scene, 0x0000FF, 0.5);


	depthButton.innerHTML = "Add Depth";
	depthButton.disabled = false;
}

let xrSession = null;
let captureCount = 0;

function onXrButtonClicked() {
	if (!xrSession) {
		activateXR();
		document.getElementById('xr-button').innerText = 'Exit AR';
	}
	else {
		deativateXR();
		document.getElementById('xr-button').innerText = 'Enter AR';
		
		const depthButton = document.getElementById('depth-button');
		depthButton.innerHTML = "Add Depth";
		depthButton.disabled = false;
	}
}

async function activateXR() {
	// Add a canvas element and initialize a WebGL context that is compatible with WebXR.
	const canvas = document.createElement("canvas");
	document.body.appendChild(canvas);
	const gl = canvas.getContext("webgl", {xrCompatible: true});

	// Create a scene.
	const scene = new THREE.Scene();

	const directionalLight = new THREE.DirectionalLight(0xffffff, 0.7);
	directionalLight.position.set(10, 15, 10);
	scene.add(directionalLight);

	// Set up the WebGLRenderer, which handles rendering to the session's base layer.
	const renderer = new THREE.WebGLRenderer({
		alpha: true,
		preserveDrawingBuffer: true,
		canvas: canvas,
		context: gl
	});
	renderer.autoClear = false;
	
	// The API directly updates the camera matrices.
	// Disable matrix auto updates so three.js doesn't attempt
	// to handle the matrices independently.
	const camera = new THREE.PerspectiveCamera();
	camera.matrixAutoUpdate = false;

	// Initialize a WebXR session using "immersive-ar".
	const session = await navigator.xr.requestSession("immersive-ar", {
		requiredFeatures: ['hit-test', 'local-floor'],
		optionalFeatures: ['camera-access', 'dom-overlay'],
		domOverlay: {root: document.getElementById('overlay')}});
	xrSession = session;
	session.updateRenderState({
	baseLayer: new XRWebGLLayer(session, gl)
	});

	// A 'local' reference space has a native origin that is located
	// near the viewer's position at the time the session was created.
	const referenceSpace = await session.requestReferenceSpace('local-floor');

	// Create another XRReferenceSpace that has the viewer as the origin.
	const viewerSpace = await session.requestReferenceSpace('viewer');
	// Perform hit testing using the viewer as origin.
	const hitTestSource = await session.requestHitTestSource({ space: viewerSpace });


	// Load GLTF models.
	const loader = new GLTFLoader();
	let reticle;
	loader.load("./models/reticle/reticle.gltf", function(gltf) {
		reticle = gltf.scene;
		reticle.visible = false;
		scene.add(reticle);
	})

	let model;
	loader.load("./models/astronaut/astronaut.glb", function(gltf) {
		model = gltf.scene;
	});

	var depthFlag = false;

	const depthButton = document.getElementById('depth-button');
	depthButton.addEventListener("click", function() {
		depthFlag = true;
	});

	// session.addEventListener("select", function() {
	// 	if (model) {
	// 		const clone = model.clone();
	// 		clone.position.copy(reticle.position);
	// 		scene.add(clone);
	// 	}
	// });
	
	const addButton = document.getElementById('add-button');
	addButton.addEventListener("click", function() {
		if (model) {
			const clone = model.clone();
			clone.position.copy(reticle.position);
			scene.add(clone);
		}
	});
	

	// Track the previous frame time.
	var lastTick = 0;

	// Create a render loop that allows us to draw on the AR view.
	const onXRFrame = (time, frame) => {
		// Update time variables.
		const delta = time - lastTick;
		lastTick = time;

		// Queue up the next draw request.
		session.requestAnimationFrame(onXRFrame);
	
		// Bind the graphics framebuffer to the baseLayer's framebuffer
		gl.bindFramebuffer(gl.FRAMEBUFFER, session.renderState.baseLayer.framebuffer)
	
		// Retrieve the pose of the device.
		// XRFrame.getViewerPose can return null while the session attempts to establish tracking.
		const pose = frame.getViewerPose(referenceSpace);
		if (pose) {
			// In mobile AR, we only have one view.
			const view = pose.views[0];
		
			const viewport = session.renderState.baseLayer.getViewport(view);
			renderer.setSize(viewport.width, viewport.height)
		
			// Use the view's transform matrix and projection matrix to configure the THREE.camera.
			camera.matrix.fromArray(view.transform.matrix)
			camera.projectionMatrix.fromArray(view.projectionMatrix);
			camera.updateMatrixWorld(true);
		
			// Drawing a targeting reticle.
			const hitTestResults = frame.getHitTestResults(hitTestSource);
			if (hitTestResults.length > 0 && reticle) {
				const hitPose = hitTestResults[0].getPose(referenceSpace);
				reticle.visible = true;
				reticle.position.set(hitPose.transform.position.x, hitPose.transform.position.y, hitPose.transform.position.z);
				reticle.updateMatrixWorld(true);
			}

			const second = onSecondFirstFrame(time, delta);

			// if (second % 20 == 5) {
			if (depthFlag == true) {
				depthFlag = false;
				addDepthPointCloud(gl, session, referenceSpace, scene, camera, frame);
			}


			// Render the scene with THREE.WebGLRenderer.
			renderer.render(scene, camera);
		}
	}
	session.requestAnimationFrame(onXRFrame);
}

function deativateXR() {
	xrSession.end();
	xrSession = null;
	captureCount = 0;
}

window.onXrButtonClicked = onXrButtonClicked;