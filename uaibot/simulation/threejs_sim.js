import {
	Object3D, Vector3, BoxBufferGeometry, Color, Mesh, MeshBasicMaterial, PerspectiveCamera, OrthographicCamera,
	Scene, WebGLRenderer, AmbientLight, DirectionalLight, HemisphereLight, MeshStandardMaterial,
	AxesHelper, GridHelper, Matrix4, SphereBufferGeometry, CylinderBufferGeometry, ConeGeometry, Group, LoadingManager, MeshPhysicalMaterial, Vector2, FrontSide,
	BackSide, DoubleSide, PMREMGenerator, TextureLoader, PointLight, UVMapping, CubeReflectionMapping, CubeRefractionMapping,
	EquirectangularReflectionMapping, EquirectangularRefractionMapping, CubeUVReflectionMapping,
	CubeUVRefractionMapping, RepeatWrapping, ClampToEdgeWrapping, MirroredRepeatWrapping,
	NearestFilter, LinearFilter, NearestMipmapNearestFilter, NearestMipmapLinearFilter, LinearMipmapNearestFilter,
	LinearMipmapLinearFilter, BufferGeometry, Float32BufferAttribute, PointsMaterial, Points, ArrowHelper, TextGeometry, CubeTextureLoader, sRGBEncoding
} from 'https://cdn.skypack.dev/three@0.135.0/build/three.module.js';
import { OrbitControls } from 'https://cdn.skypack.dev/three@0.135.0/examples/jsm/controls/OrbitControls.js';
import { OBJLoader } from 'https://cdn.skypack.dev/three@0.135.0/examples/jsm/loaders/OBJLoader.js';
import { STLLoader } from 'https://cdn.skypack.dev/three@0.135.0/examples/jsm/loaders/STLLoader.js';
import { MTLLoader } from 'https://cdn.skypack.dev/three@0.135.0/examples/jsm/loaders/MTLLoader.js';
import { GLTFLoader } from 'https://cdn.skypack.dev/three@0.135.0/examples/jsm/loaders/GLTFLoader.js';
import { ColladaLoader } from 'https://cdn.skypack.dev/three@0.135.0/examples/jsm/loaders/ColladaLoader.js';
import { GUI } from 'https://cdn.skypack.dev/dat.gui';
import { ParametricGeometry } from 'https://cdn.skypack.dev/three@0.135.0/examples/jsm/geometries/ParametricGeometry.js'


//--------------------SIMULATION ELEMENTS---------------------

//SIMULATION PARAMETERS GO HERE

function elapsedMs() {
	return (new Date()).getTime();
}


class Objsim {
	constructor(_frames) {
		this.frames = _frames;
		this.currentFrame = 0;
		this.shape = "I HAVE NO SHAPE YET";
	}

	hasLoaded() {
		return true;
	}

	nextFrame() {
		while ((this.currentFrame < this.frames.length - 1) && (1000 * this.frames[this.currentFrame + 1][0] + delay < (elapsedMs() - startMs))) {
			this.currentFrame += 1;
		}
	}
	setFrameTime(time) {
		this.currentFrame = 0;
		while ((this.currentFrame < this.frames.length - 1) && (this.frames[this.currentFrame + 1][0] < time)) {
			this.currentFrame += 1;
		}
	}
	showFrame() {
		this.shape.matrix.set(this.frames[this.currentFrame][1], this.frames[this.currentFrame][2], this.frames[this.currentFrame][3], this.frames[this.currentFrame][4],
			this.frames[this.currentFrame][5], this.frames[this.currentFrame][6], this.frames[this.currentFrame][7], this.frames[this.currentFrame][8],
			this.frames[this.currentFrame][9], this.frames[this.currentFrame][10], this.frames[this.currentFrame][11], this.frames[this.currentFrame][12],
			this.frames[this.currentFrame][13], this.frames[this.currentFrame][14], this.frames[this.currentFrame][15], this.frames[this.currentFrame][16]);


	}
	setEnvMap(envMap) {
		this.shape.material.envMap = envMap;
		this.shape.material.needsUpdate = true;
	}
}

class Box extends Objsim {
	constructor(_width, _height, _depth, _frames, _material) {
		super(_frames);
		this.width = _width;
		this.height = _height;
		this.depth = _depth;
		const geometry = new BoxBufferGeometry(this.width, this.height, this.depth);
		const cube = new Mesh(geometry, _material);
		cube.matrixAutoUpdate = false;
		this.shape = cube;
	}
}

class Ball extends Objsim {
	constructor(_radius, _frames, _material) {
		super(_frames);
		this.radius = _radius;
		const geometry = new SphereBufferGeometry(this.radius, 64, 32);
		const sphere = new Mesh(geometry, _material);
		sphere.matrixAutoUpdate = false;
		this.shape = sphere;
	}
}

class Cylinder extends Objsim {
	constructor(_radius, _height, _frames, _material) {
		super(_frames);
		this.radius = _radius;
		this.height = _height;
		const geometry = new CylinderBufferGeometry(this.radius, this.radius, this.height, 20);
		const cylinder = new Mesh(geometry, _material);
		cylinder.matrixAutoUpdate = false;
		this.shape = cylinder;
	}
}

class Cone extends Objsim {
	constructor(_radius, _height, _frames, _material) {
		super(_frames);
		this.radius = _radius;
		this.height = _height;
		const geometry = new ConeGeometry(this.radius, this.height, 20, 5, false, 0.0, 6.28);
		const cone = new Mesh(geometry, _material);
		cone.matrixAutoUpdate = false;
		this.shape = cone;
	}
}

function signFun(x, n) {
	return Math.sign(x) * (Math.pow(Math.abs(x), n))
}

class SmoothBox extends Objsim {



	constructor(_width, _height, _depth, _frames, _material) {
		super(_frames);
		this.width = _width;
		this.height = _height;
		this.depth = _depth;

		const geometry = new ParametricGeometry(function fun(u, v, target) {
			let cosu = signFun(Math.cos(0.0001 * Math.PI + 0.9999 * Math.PI * u), 0.5)
			let sinu = signFun(Math.sin(0.0001 * Math.PI + 0.9999 * Math.PI * u), 0.5)
			let cosv = signFun(Math.cos(2 * Math.PI * v), 0.5)
			let sinv = signFun(Math.sin(2 * Math.PI * v), 0.5)

			let x = (_width / 2) * sinu * cosv;
			let y = (_height / 2) * sinu * sinv;
			let z = (_depth / 2) * cosu;

			target.set(x, y, z);

		}, 25, 25);


		const smoothCube = new Mesh(geometry, _material);
		smoothCube.matrixAutoUpdate = false;
		this.shape = smoothCube;
	}



}

class ConvexPolytope extends Objsim {



	constructor(_vertexes, _faces, _frames, _material) {
		super(_frames);
		this.vertexes = _vertexes;
		this.faces = _faces;


		const geometry = new BufferGeometry();
		const verticesArray = [];
		const indicesArray = [];

		// this.vertexes.forEach(v => {
		// 	verticesArray.push(v[0], v[1], v[2]);
		// });

		this.vertexes.forEach(v => {
			verticesArray.push(v[0], v[2], -v[1]);
		});

		this.faces.forEach(f => {
			indicesArray.push(f[0], f[2], f[1]);
		});

		// this.faces.forEach(f => {
		// 	indicesArray.push(f[0], f[1], f[2]);
		// });


		geometry.setAttribute(
			"position",
			new Float32BufferAttribute(verticesArray, 3)
		);
		geometry.setIndex(indicesArray);
		geometry.computeVertexNormals();

		const polytope = new Mesh(geometry, _material);
		polytope.matrixAutoUpdate = false;
		this.shape = polytope;
	}



}

class PointCloud extends Objsim {

	constructor(_points, _frames, _color, _size) {
		super(_frames);

		this.color = _color;
		this.points = _points
		this.size = _size;

		const vertices = []
		const geometry = new BufferGeometry();
		geometry.setAttribute('position', new Float32BufferAttribute(vertices, 3));
		const material = new PointsMaterial({ color: this.color, size: this.size });
		this.shape = new Points(geometry, material);
	}

	showFrame() {
		const vertices = [];

		for (let i = this.frames[this.currentFrame][1]; i < this.frames[this.currentFrame][2]; i++) {
			vertices.push(this.points[0][i], this.points[1][i], this.points[2][i]);
		}

		this.shape.geometry.setAttribute('position', new Float32BufferAttribute(vertices, 3));

	}

	setEnvMap(envMap) { }

}

class RigidObject extends Objsim {

	constructor(_frames, _list_3dobject) {
		super(_frames);

		this.shape = new Group();
		this.shape.matrixAutoUpdate = false;

		const manager = new LoadingManager();

		let loader;

		for (let i = 0; i < _list_3dobject.length; i++) {
			const _3dobject = _list_3dobject[i];
			if (_3dobject.type === 'obj') {
				loader = new OBJLoader(manager);
				loader.load(_3dobject.url, (root) => {

					root.scale.set(_3dobject.scale, _3dobject.scale, _3dobject.scale);
					root.applyMatrix4(_3dobject.matrix)

					root.traverse(function (child) {
						if (child instanceof Mesh) {
							child.material = _3dobject.mesh_material;
						}
					});

					this.shape.add(root);
				});
			}
			if (_3dobject.type === 'stl') {
				loader = new STLLoader(manager);
				loader.load(_3dobject.url, (root) => {
					const mesh = new Mesh(root);

					mesh.scale.set(_3dobject.scale, _3dobject.scale, _3dobject.scale);
					mesh.applyMatrix4(_3dobject.matrix)

					mesh.traverse(function (child) {
						if (child instanceof Mesh) {
							child.material = _3dobject.mesh_material;
						}
					});

					this.shape.add(mesh);
				});
			}
			if (_3dobject.type === 'dae') {
				loader = new ColladaLoader(manager);
				loader.load(_3dobject.url, (root) => {
					const group = root.scene;

					group.scale.set(_3dobject.scale, _3dobject.scale, _3dobject.scale);
					group.applyMatrix4(_3dobject.matrix)

					group.traverse(function (child) {
						if (child instanceof Mesh) {
							child.material = _3dobject.mesh_material;
						}
					});

					this.shape.add(group);
				});
			}
		}

	}

	setEnvMapTemp(variable, envMap) {
		if (variable instanceof Mesh) {
			variable.material.envMap = envMap;
			variable.material.needsUpdate = true;
		}
		else if (variable instanceof Group) {
			for (let i = 0; i < variable.children.length; i++) {
				this.setEnvMapTemp(variable.children[i]);
			}
		}
	}
	setEnvMap(envMap) {
		this.setEnvMapTemp(this.shape, envMap);
	}

}

class Vector extends Objsim {


	constructor(_frames, _color, _thickness, _origin, _dir, _length) {
		super(_frames);

		this.color = _color;
		this.thickness = _thickness
		this.origin = new Vector3(_origin[0], _origin[1], _origin[2]);
		this.dir = new Vector3(_dir[0], _dir[1], _dir[2]);
		this._length = length

		this.shape = new ArrowHelper(this.dir, this.origin, this.length, this.color);
		this.shape.line.material.linewidth = this.thickness;

	}

	showFrame() {
		this.shape.setLength(this.frames[this.currentFrame][3]);
		const dir = this.frames[this.currentFrame][2];
		const origin = this.frames[this.currentFrame][1];

		this.shape.setDirection(new Vector3(dir[0], dir[1], dir[2]));
		this.shape.position.set(origin[0], origin[1], origin[2]);
	}
	setEnvMap(envMap) { }
}

class PointLightSim extends Objsim {


	constructor(_frames) {
		super(_frames);
		this.shape = new PointLight(this.frames[0][1], this.frames[0][2], this.frames[0][3]);
		this.shape.position.set(this.frames[0][4][0], this.frames[0][4][1], this.frames[0][4][2]);
	}

	showFrame() {
		this.shape.color.set(this.frames[this.currentFrame][1]);
		this.shape.intensity = this.frames[this.currentFrame][2];
		this.shape.distance = this.frames[this.currentFrame][3];
		this.shape.position.set(this.frames[this.currentFrame][4][0], this.frames[this.currentFrame][4][1], this.frames[this.currentFrame][4][2]);
	}
	setEnvMap(envMap) { }
}

class HTMLDiv extends Objsim {

	constructor(_id, _frames) {
		super(_frames);
		this.id = _id;
		this.shape = undefined;
	}

	showFrame() {
		var element = document.getElementById(this.id);

		element.innerHTML = this.frames[this.currentFrame][1];
		element.style.position = "absolute";
		element.style = this.frames[this.currentFrame][2];
	}
	setEnvMap(envMap) { }
}

class Robot extends Objsim {

	constructor(_objBase, _link, _frames, _htm_base_0, _htm_n_eef, _eef_frame_visible) {
		super(_frames);

		this.loadedObjs = 0
		this.links = _link;

		this.totalNumberOfObjects = _objBase.length;

		for (let i = 0; i < this.links.length; i++) {
			this.totalNumberOfObjects += this.links[i].model3d.length;
		}

		this.htm_base_0 = _htm_base_0;

		//Function that creates a generic robot
		const base = new Group();
		base.name = "base";

		const baseFrame = new Group();
		baseFrame.name = "baseFrame";
		baseFrame.applyMatrix4(this.htm_base_0);


		//Create link groups
		let linkGroupList = [];

		for (let i = 0; i <= this.links.length; i++) {

			var linkGroup = new Group();
			linkGroup.name = "linkFrame" + i.toString();
			linkGroupList.push(linkGroup);

			linkGroup = new Group();
			linkGroup.name = "link" + i.toString();
			//var axesHelper = new AxesHelper(0.2);
			//axesHelper.matrixAutoUpdate = false;
			//axesHelper.visible = true;
			//linkGroup.add(axesHelper)

			if (i < this.links.length) {
				linkGroup.rotateZ(this.links[i].theta);
				linkGroup.translateZ(this.links[i].d);
				linkGroup.rotateX(this.links[i].alpha);
				linkGroup.translateX(this.links[i].a);
			}


			linkGroup.updateMatrix();

			linkGroupList.push(linkGroup);
		}

		var linkGroup = new Group();
		linkGroup.name = "eefFrame";
		linkGroupList.push(linkGroup);

		linkGroup = new Group();
		linkGroup.name = "eef";
		// var axesHelper = new AxesHelper(0.2);
		// axesHelper.matrixAutoUpdate = false;
		// axesHelper.visible = _eef_frame_visible;
		// linkGroup.add(axesHelper);
		linkGroup.applyMatrix4(_htm_n_eef)
		linkGroup.updateMatrix();
		linkGroupList.push(linkGroup);


		base.add(baseFrame)
		baseFrame.add(linkGroupList[0])

		for (let i = 0; i < linkGroupList.length - 1; i++) {
			linkGroupList[i].add(linkGroupList[i + 1])
		}

		//Load 3D models

		this.shape = base;
		this.shape.matrixAutoUpdate = false;

		this.loadObj(_objBase, "base", "base", false)
		for (let i = 0; i < this.links.length - 1; i++) {
			this.loadObj(this.links[i].model3d, "link" + (i + 1).toString(), "link" + (i).toString(), this.links[i].showFrame)
		}
		this.loadObj(this.links[this.links.length - 1].model3d, "link" + (this.links.length).toString(), "link" + (this.links.length).toString(), false)

		this.delta_config = []

		for (let i = 0; i < this.links.length; i++) {
			if (this.links[i].jointType == 0) {
				this.delta_config.push(this.shape.getObjectByName("link" + (i).toString()).rotation.z)
			}
			if (this.links[i].jointType == 1) {
				this.delta_config.push(this.shape.getObjectByName("link" + (i).toString()).position.z)

			}
		}

	}

	loadObj(obj, name1, name2, axisVisible) {

		const manager = new LoadingManager();
		const objLoader = new OBJLoader(manager);
		const stlLoader = new STLLoader(manager);
		const colladaLoader = new ColladaLoader(manager);
		const mtlLoader = new MTLLoader(manager);
		const gltfloader = new GLTFLoader(manager);


		for (let i = 0; i < obj.length; i++) {


			if (obj[i].type == "obj") {
				if (obj[i].mesh_material.type == 'mtl') {
					mtlLoader.load(obj[i].mesh_material.url, (materials) => {
						materials.preload();

						objLoader.setMaterials(materials);
						objLoader.load(obj[i].url, (root) => {
							root.scale.set(obj[i].scale, obj[i].scale, obj[i].scale);
							root.applyMatrix4(obj[i].matrix);

							// Override opacity if specified
							const userOpacity = obj[i].mesh_material?.opacity;
							if (userOpacity !== undefined) {
								root.traverse((child) => {
									if (child instanceof Mesh && child.material) {
										if (Array.isArray(child.material)) {
											child.material.forEach((mat) => {
												if (mat) {
													mat.opacity = userOpacity;
													mat.transparent = userOpacity < 1.0;
													mat.needsUpdate = true;
												}
											});
										} else {
											child.material.opacity = userOpacity;
											child.material.transparent = userOpacity < 1.0;
											child.material.needsUpdate = true;
										}
									}
								});
							}


							this.shape.getObjectByName(name2).add(root);
							this.loadedObjs += 1;
						});
					});
				} else {
					objLoader.load(obj[i].url, (root) => {
						root.scale.set(obj[i].scale, obj[i].scale, obj[i].scale);
						root.applyMatrix4(obj[i].matrix);

						root.traverse((child) => {
							if (child instanceof Mesh) {
								child.material = obj[i].mesh_material;
							}
						});

						this.shape.getObjectByName(name2).add(root);
						this.loadedObjs += 1;
					});
				}
			}
			if (obj[i].type == "stl") {
				stlLoader.load(obj[i].url, (root) => {
					const mesh = new Mesh(root);
					mesh.scale.set(obj[i].scale, obj[i].scale, obj[i].scale);
					mesh.applyMatrix4(obj[i].matrix)

					mesh.traverse(function (child) {
						if (child instanceof Mesh) {
							child.material = obj[i].mesh_material;
						}
					});

					this.shape.getObjectByName(name2).add(mesh);
					this.loadedObjs += 1
				});
			}
			if (obj[i].type == "dae") {
				colladaLoader.load(obj[i].url, (root) => {
					const group = root.scene;
					group.scale.set(obj[i].scale, obj[i].scale, obj[i].scale);
					group.applyMatrix4(obj[i].matrix)

					group.traverse(function (child) {
						if (child instanceof Mesh) {
							child.material = obj[i].mesh_material;
						}
					});

					this.shape.getObjectByName(name2).add(group);
					this.loadedObjs += 1
				});
			}
			if (obj[i].type == "glb") {
				gltfloader.load(obj[i].url, (root) => {
					const group = root.scene;
					group.scale.set(obj[i].scale, obj[i].scale, obj[i].scale);
					group.applyMatrix4(obj[i].matrix);

					if (obj[i].mesh_material.type !== 'glb') {
						group.traverse(function (child) {
							if (child instanceof Mesh) {
								child.material = obj[i].mesh_material;
							}
						});
					}
					else {
						const userOpacity = obj[i].mesh_material?.opacity;
						if (userOpacity !== undefined) {
							group.traverse((child) => {
								if (child instanceof Mesh && child.material) {
									if (Array.isArray(child.material)) {
										child.material.forEach((mat) => {
											if (mat) {
												mat.opacity = userOpacity;
												mat.transparent = userOpacity < 1.0;
												mat.needsUpdate = true;
												mat.material.format = 1023;
											}
										});
									} else {
										child.material.opacity = userOpacity;
										child.material.transparent = userOpacity < 1.0;
										child.material.needsUpdate = true;
										child.material.format = 1023;
									}
								}
							});
						}
					}

					this.shape.getObjectByName(name2).add(group);
					this.loadedObjs += 1;
				});
			}

		}

		// this.shape.getObjectByName(name1).getObjectByProperty("type", "AxesHelper").visible = axisVisible;
	}

	hasLoaded() {
		return this.loadedObjs == this.totalNumberOfObjects;
	}

	//Function that updates frames
	showFrame() {
		//setting robot position
		this.shape.matrix.set(this.frames[this.currentFrame][1], this.frames[this.currentFrame][2], this.frames[this.currentFrame][3], this.frames[this.currentFrame][4],
			this.frames[this.currentFrame][5], this.frames[this.currentFrame][6], this.frames[this.currentFrame][7], this.frames[this.currentFrame][8],
			this.frames[this.currentFrame][9], this.frames[this.currentFrame][10], this.frames[this.currentFrame][11], this.frames[this.currentFrame][12],
			this.frames[this.currentFrame][13], this.frames[this.currentFrame][14], this.frames[this.currentFrame][15], this.frames[this.currentFrame][16]);
		//setting robot configuration

		if (this.frames[this.currentFrame][17] != undefined) {

			for (let i = 0; i < this.links.length; i++) {
				const linkName = "linkFrame" + (i).toString();

				if (this.links[i].jointType == 0) {
					this.shape.getObjectByName(linkName).rotation.z = this.frames[this.currentFrame][17][i] + this.delta_config[i];
				} else if (this.links[i].jointType == 1) {
					this.shape.getObjectByName(linkName).position.z = this.frames[this.currentFrame][17][i] + this.delta_config[i];
				}

			}
		}
	}

	setEnvMapTemp(variable, envMap) {
		if (variable instanceof Mesh) {
			variable.material.envMap = envMap;
			variable.material.needsUpdate = true;
		}
		else if (variable instanceof Group) {
			for (let i = 0; i < variable.children.length; i++) {
				this.setEnvMapTemp(variable.children[i], envMap);
			}
		}
	}
	setEnvMap(envMap) {
		this.setEnvMapTemp(this.shape, envMap);
	}
}

class Curve extends Objsim {
    constructor(_frames, _color, _size) {
        // _frames: array of [time, flatPointsArray]
        // flatPointsArray is a flat array of numbers: [x0,y0,z0, x1,y1,z1, ...]
        super(_frames);

        this.color = _color;
        this.size = _size;
        this.frames = _frames; // already set in super

        // Determine number of points from the first frame (if any)
        if (this.frames.length > 0) {
            const firstFlat = this.frames[0][1];
            this.numPoints = firstFlat.length / 3;
        } else {
            this.numPoints = 0;
        }

        // Create geometry and material
        const geometry = new BufferGeometry();
        const material = new PointsMaterial({ color: this.color, size: this.size });
        this.shape = new Points(geometry, material);
    }

    showFrame() {
        // currentFrame is updated by Objsim's nextFrame() or setFrameTime()
        const flatPoints = this.frames[this.currentFrame][1];
        this.shape.geometry.setAttribute('position', new Float32BufferAttribute(flatPoints, 3));
    }

    setEnvMap(envMap) { }
}
//------------------------------------------------------------

//--------------- BASIC ELEMENTS OF ANY SCENE ----------------

Object3D.DefaultUp = new Vector3(0, 0, 1); //Pointing Z axis up
const canvas = document.getElementById('scene_' + sceneID);// Selecting canvas

const scene = new Scene();//Instantiate the Scene
scene.background = new Color(backgroundColor);//Set background color

let camera;

if (cameraType == "perspective") {
	camera = new PerspectiveCamera(60, canvas.clientWidth / canvas.clientHeight, 0.1, 100);//Instantiate a camera
}
if (cameraType == "orthographic") {
	camera = new OrthographicCamera(-3.2, 3.2, 2.4, -2.4, 0.01, 100);
}



var ambientLight = new HemisphereLight('white', 'darkslategrey', 3);//Instantiate Ambient light
ambientLight.intensity = ambientLightIntensity;
scene.add(ambientLight);

const controls = new OrbitControls(camera, canvas);	//Instantiate orbit controls
//controls.target.set(0, 0, 0);//Point camera at the origin

camera.position.set(cameraStartPose[0], cameraStartPose[1], cameraStartPose[2]);
//camera.rotation.set(cameraStartPose[3], cameraStartPose[4], cameraStartPose[5]);
camera.lookAt(cameraStartPose[3], cameraStartPose[4], cameraStartPose[5]);
camera.zoom = cameraStartPose[6];
camera.updateProjectionMatrix();

const renderer = new WebGLRenderer({ canvas, antialias: true });//Instantiate renderer
renderer.physicallyCorrectLights = true;//Enable physically Correct Lights
renderer.setSize(canvas.clientWidth, canvas.clientHeight);//Set render size
renderer.setPixelRatio(window.devicePixelRatio);//Set pixel ratio

let sceneElements = [];

// if (showWorldFrame) {
// 	const axesHelper = new AxesHelper(0.5);
// 	axesHelper.renderOrder = 1;
// 	scene.add(axesHelper);
// }

if (showGrid) {
	const gridHelper = new GridHelper(6, 12);
	scene.add(gridHelper);
	gridHelper.rotation.x = 3.14 / 2;
}





//------------------------------------------------------------

//--------------- ADDING ELEMENTS TO THIS SCENE ---------------
//USER INPUT GOES HERE



// add stuff to the scene
for (let i = 0; i < sceneElements.length; i++) {
	if (typeof sceneElements[i].shape != "undefined") {
		scene.add(sceneElements[i].shape);
	}
}



//Add LDR to the scene
if (!(ldrUrls.length == 0)) {

	const pmremGenerator = new PMREMGenerator(renderer);
	var ldrCubeMap = new CubeTextureLoader().load(ldrUrls, function () {

	    ldrCubeMap.encoding = sRGBEncoding;
	    const ldrCubeRenderTarget = pmremGenerator.fromCubemap(ldrCubeMap);
	    const envMap = ldrCubeRenderTarget.texture;

	    scene.environment = envMap;

	    for (let i = 0; i < sceneElements.length; i++) {
		const sceneObj = sceneElements[i];

		sceneObj.setEnvMap(envMap);

	    }

	    scene.background = ldrCubeMap;
	});

}



//Add gui
var gui = new GUI({ autoPlace: false, width: "800px" });
var customContainer = document.getElementById('canvas_container_' + sceneID);
customContainer.appendChild(gui.domElement);

var automaticPlay = true;
var manualTime = 0;

gui.add({ Progress: 0 }, "Progress", 0, maxTime, 0.01).onChange(function (value) {
	automaticPlay = false;
	manualTime = value + 0.0001;
	btn.innerHTML = "&#9654";
})
customContainer.getElementsByClassName('c')[0].style.width = (canvas.clientWidth - 20).toString() + 'px';
customContainer.getElementsByClassName('dg main')[0].style.width = canvas.clientWidth.toString() + 'px';
customContainer.getElementsByClassName('dg main')[0].style.right = '0px'; //0
customContainer.getElementsByClassName('dg main')[0].style.top = '0px'; //550
customContainer.getElementsByClassName('dg main')[0].style.position = 'relative'; //absolute
customContainer.getElementsByClassName('close-button close-bottom')[0].style.height = '0px';
customContainer.getElementsByClassName('close-button close-bottom')[0].innerHTML = '';
customContainer.getElementsByClassName('property-name')[0].innerHTML = '';
customContainer.getElementsByClassName('slider')[0].style.width = (canvas.clientWidth - 100).toString() + 'px'
customContainer.getElementsByClassName('slider-fg')[0].style.backgroundColor = '#19bd39'
customContainer.querySelectorAll("input[type=text]")[0].style.width = '50px';
customContainer.querySelectorAll("input[type=text]")[0].style.color = '#19bd39';

customContainer.getElementsByClassName('dg')[0].style.borderLeft = '4px solid #1a1a1a'
customContainer.getElementsByClassName('cr')[0].style.borderLeft = '4px solid #1a1a1a'
customContainer.getElementsByClassName('number')[0].style.borderLeft = '4px solid #1a1a1a'
var recoveryCurrentTime = false;

let btn = document.createElement("button");
btn.style.backgroundColor = "#1a1a1a";
btn.style.color = "#19bd39";
btn.style.border = "none";
btn.style.height = "28px";
btn.style.width = "28px";
btn.innerHTML = "||";
btn.onclick = function () {

	automaticPlay = !automaticPlay;

	if (automaticPlay) {
		btn.innerHTML = "||";
		recoveryCurrentTime = true;
	}
	else {
		manualTime = parseFloat(customContainer.querySelectorAll("input[type=text]")[0].value) + 0.0001;
		btn.innerHTML = "&#9654";
	}

};
customContainer.getElementsByClassName('c')[0].style.verticalAlign = "middle"
customContainer.getElementsByClassName('c')[0].appendChild(btn);
//------------------------------------------------------------

//Wait until all objects loaded

var loadedAll = false;
var sceneDrawn = false;


//Get starting ms
var startMs = elapsedMs()

//Initialize
var heightScene;
var heightBar;

document.getElementsByClassName('controller')[0].appendChild(customContainer.getElementsByClassName('dg main')[0])
document.getElementsByClassName('controller')[0].style.opacity = "0"

heightScene = document.getElementById('scene_' + sceneID).style.height;
document.getElementById('scene_' + sceneID).style.visibility = 'hidden';
document.getElementById('scene_' + sceneID).style.height = '0px';

heightBar = customContainer.getElementsByClassName('dg main')[0].style.height;
customContainer.getElementsByClassName('dg main')[0].style.visibility = 'hidden';
customContainer.getElementsByClassName('dg main')[0].style.height = '0px';


//-------------------- THE ANIMATION LOOP -------------------



renderer.setAnimationLoop(() => {


	loadedAll = true;

	for (let i = 0; i < sceneElements.length; i++) {
		loadedAll = loadedAll && sceneElements[i].hasLoaded();
	}

	if (!loadedAll) {
		startMs = elapsedMs();
	}
	else {
		//document.getElementById('canvas_container').style.visibility = 'visible';
		if ((elapsedMs() - startMs > delay) && !sceneDrawn) {
			document.getElementById('scene_' + sceneID).style.visibility = 'visible';
			document.getElementById('canvas_container_' + sceneID).getElementsByClassName('dg main')[0].style.visibility = 'visible';
			document.getElementById('scene_' + sceneID).style.height = heightScene;
			document.getElementById('canvas_container_' + sceneID).getElementsByClassName('dg main')[0].style.height = heightBar;
			document.getElementById('loading_screen_' + sceneID).remove();
			sceneDrawn = true;
		}
	}


	//MAGIC HAPPENS HERE!!!
	if (loadedAll && automaticPlay) {

		var currentSeconds;
		if (!recoveryCurrentTime) {

			for (let i = 0; i < sceneElements.length; i++) {
				sceneElements[i].nextFrame();
				sceneElements[i].showFrame();
			}

			currentSeconds = Math.min(0.001 * Math.max(elapsedMs() - startMs - delay, 0), maxTime);

			var percentage = (100 * currentSeconds / (0.000001 + maxTime)).toString();
			customContainer.getElementsByClassName('slider-fg')[0].style.width = percentage + "%";
			customContainer.querySelectorAll("input[type=text]")[0].value = Math.round(100 * currentSeconds) / 100;
		}
		else {
			currentSeconds = customContainer.querySelectorAll("input[type=text]")[0].value;
			startMs = elapsedMs() - delay - 1000 * currentSeconds;
			recoveryCurrentTime = false;
		}

	}
	if (loadedAll && !automaticPlay) {
		for (let i = 0; i < sceneElements.length; i++) {
			sceneElements[i].setFrameTime(manualTime);
			sceneElements[i].showFrame();
		}
	}

	renderer.render(scene, camera);
});
//------------------------------------------------------------


// window.addEventListener('resize', onWindowResize, false);

// function onWindowResize() {
    
//     const canvas = document.querySelector('canvas');
//     const widthAttr = canvas.getAttribute('width');
//     const heightAttr = canvas.getAttribute('height');
//     const factor = widthAttr/heightAttr
// 	const canvas_div =  document.getElementById('canvas_container_'+sceneID)
    
// 	if(factor>1)
// 	{
// 		const width = Math.round(0.9*factor*window.innerHeight);
// 		const height = 0.9*window.innerHeight;
		
// 		canvas_div.style.width = width;
//     	canvas_div.style.height = height;
			
// 		camera.aspect = width / height;
// 		camera.updateProjectionMatrix();
// 		renderer.setSize(width, height);
// 	}
// 	else
// 	{
// 		const width = 0.9*window.innerWidth;
// 		const height =  Math.round((0.9/factor)*window.innerWidth);
		
// 		canvas_div.style.width = width;
//     	canvas_div.style.height = height;
			
// 		camera.aspect = width / height;
// 		camera.updateProjectionMatrix();
// 		renderer.setSize(width, height);
// 	}


//     // ---- GUI WIDTH FIX ----
//     const canvasWidth = renderer.domElement.clientWidth;
//     const customContainer = document.getElementById('canvas_container_' + sceneID);

//     if (customContainer) {
//         try {
//             customContainer.getElementsByClassName('c')[0].style.width = (canvasWidth - 20) + 'px';
//             customContainer.getElementsByClassName('dg main')[0].style.width = canvasWidth + 'px';
//             customContainer.getElementsByClassName('slider')[0].style.width = (canvasWidth - 100) + 'px';
//         } catch (e) {
//             console.warn('GUI resize skipped:', e);
//         }
//     }
// }
