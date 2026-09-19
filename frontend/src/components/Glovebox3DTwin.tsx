import React, { useRef, useEffect } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { Attitude3D } from '../types/detections';

interface Glovebox3DTwinProps {
  activeStep: number;
  attitude?: Attitude3D;
  className?: string;
}

export const Glovebox3DTwin: React.FC<Glovebox3DTwinProps> = ({ activeStep, attitude, className }) => {
  const mountRef = useRef<HTMLDivElement>(null);
  const activeStepRef = useRef(activeStep);
  const attitudeRef = useRef(attitude);

  useEffect(() => {
    activeStepRef.current = activeStep;
  }, [activeStep]);

  useEffect(() => {
    attitudeRef.current = attitude;
  }, [attitude]);

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;

    const W = mount.clientWidth;
    const H = mount.clientHeight;

    // Renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(W, H);
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setClearColor(0x050a14, 1);
    renderer.shadowMap.enabled = true;
    mount.appendChild(renderer.domElement);

    // Scene & Camera
    const scene = new THREE.Scene();
    scene.fog = new THREE.Fog(0x050a14, 8, 20);
    const camera = new THREE.PerspectiveCamera(55, W / H, 0.1, 50);
    camera.position.set(0, 1.5, 4);

    // Controls
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.minDistance = 2;
    controls.maxDistance = 8;
    controls.target.set(0, 0, 0);

    // Lights
    scene.add(new THREE.AmbientLight(0x112244, 0.5));
    const cyan_light = new THREE.PointLight(0x38bdf8, 1.5, 8);
    cyan_light.position.set(-2, 2, 2);
    scene.add(cyan_light);
    const white_light = new THREE.PointLight(0xffffff, 1.0, 10);
    white_light.position.set(2, 3, 3);
    scene.add(white_light);

    // ── Glovebox enclosure ──
    const boxGeo = new THREE.BoxGeometry(3, 2, 1.5);
    const boxMat = new THREE.MeshStandardMaterial({ color: 0x2a3a4a, roughness: 0.8, metalness: 0.4, wireframe: false });
    const boxMesh = new THREE.Mesh(boxGeo, boxMat);
    boxMesh.position.set(0, 0, 0);
    scene.add(boxMesh);

    // Glass front window
    const windowGeo = new THREE.PlaneGeometry(2.4, 1.6);
    const windowMat = new THREE.MeshPhysicalMaterial({
      color: 0x38bdf8, transparent: true, opacity: 0.08,
      roughness: 0, metalness: 0.1, side: THREE.DoubleSide,
    });
    const windowMesh = new THREE.Mesh(windowGeo, windowMat);
    windowMesh.position.set(0, 0, 0.76);
    scene.add(windowMesh);

    // Glove ports
    const portGeo = new THREE.CylinderGeometry(0.18, 0.18, 0.2, 16);
    const portMat = new THREE.MeshStandardMaterial({ color: 0x1a2a3a, roughness: 0.9 });
    const portL = new THREE.Mesh(portGeo, portMat);
    portL.rotation.z = Math.PI / 2;
    portL.position.set(-1.55, -0.4, 0);
    scene.add(portL);
    const portR = portL.clone();
    portR.position.set(1.55, -0.4, 0);
    scene.add(portR);

    // ── Centrifuge body ──
    const centGeo = new THREE.CylinderGeometry(0.4, 0.4, 0.3, 24);
    const centMat = new THREE.MeshStandardMaterial({ color: 0x3a5a6a, roughness: 0.5, metalness: 0.6 });
    const centMesh = new THREE.Mesh(centGeo, centMat);
    centMesh.position.set(0, -0.3, 0);
    scene.add(centMesh);

    // Centrifuge spokes
    const spokeMat = new THREE.MeshStandardMaterial({ color: 0x5a8a9a, roughness: 0.4, metalness: 0.7 });
    const spokes: THREE.Mesh[] = [];
    for (let i = 0; i < 4; i++) {
      const sg = new THREE.BoxGeometry(0.7, 0.04, 0.04);
      const sm = new THREE.Mesh(sg, spokeMat);
      sm.position.set(0, -0.16, 0);
      sm.rotation.y = (i * Math.PI) / 2;
      scene.add(sm);
      spokes.push(sm);
    }

    // ── Sample tubes ──
    const tubeBodyMat = new THREE.MeshStandardMaterial({ color: 0xdddddd, roughness: 0.3, transparent: true, opacity: 0.9 });
    const redCapMat   = new THREE.MeshStandardMaterial({ color: 0xf43f5e, roughness: 0.4 });
    const yelCapMat   = new THREE.MeshStandardMaterial({ color: 0xfbbf24, roughness: 0.4 });

    function makeTube(capMat: THREE.Material, initX: number) {
      const g = new THREE.Group();
      const bodyGeo = new THREE.CylinderGeometry(0.045, 0.045, 0.18, 12);
      g.add(new THREE.Mesh(bodyGeo, tubeBodyMat));
      const capGeo = new THREE.CylinderGeometry(0.05, 0.05, 0.04, 12);
      const cap = new THREE.Mesh(capGeo, capMat);
      cap.position.y = 0.11;
      g.add(cap);
      g.position.set(initX, -0.2, 0.2);
      return g;
    }

    const smpRed = makeTube(redCapMat, -0.6);
    const smpYel = makeTube(yelCapMat, 0.6);
    scene.add(smpRed);
    scene.add(smpYel);

    // ── Kinematic arm ──
    const armMat = new THREE.MeshStandardMaterial({ color: 0x4a6a8a, roughness: 0.6, metalness: 0.5 });
    const arm1Geo = new THREE.BoxGeometry(0.08, 0.5, 0.08);
    const arm1 = new THREE.Mesh(arm1Geo, armMat);
    arm1.position.set(-1.2, 0.3, 0.2);
    scene.add(arm1);
    const arm2Geo = new THREE.BoxGeometry(0.06, 0.4, 0.06);
    const arm2 = new THREE.Mesh(arm2Geo, armMat);
    arm2.position.set(-1.0, 0.1, 0.2);
    scene.add(arm2);
    const arm3Geo = new THREE.BoxGeometry(0.04, 0.25, 0.04);
    const arm3 = new THREE.Mesh(arm3Geo, armMat);
    arm3.position.set(-0.8, -0.1, 0.2);
    scene.add(arm3);

    // ── Grid floor ──
    const grid = new THREE.GridHelper(6, 12, 0x1e3a5a, 0x0e2030);
    grid.position.y = -1.2;
    scene.add(grid);

    // Animation
    let rafId: number;
    const animate = () => {
      rafId = requestAnimationFrame(animate);
      const step = activeStepRef.current;
      const t = Date.now() * 0.001;

      // Centrifuge rotation speed based on step
      const spinSpeed = step === 4 ? 0.12 : step >= 2 ? 0.02 : 0.005;
      centMesh.rotation.y += spinSpeed;
      spokes.forEach(s => { s.rotation.y += spinSpeed; });

      // Sample tube lerp toward centrifuge
      const redTarget = step >= 3 ? new THREE.Vector3(-0.12, -0.18, 0) : new THREE.Vector3(-0.6, -0.2, 0.2);
      const yelTarget = step >= 4 ? new THREE.Vector3(0.12, -0.18, 0)  : new THREE.Vector3(0.6, -0.2, 0.2);
      smpRed.position.lerp(redTarget, 0.03);
      smpYel.position.lerp(yelTarget, 0.03);

      // Kinematic arm animation & 3D Attitude Synchronization
      const att = attitudeRef.current;
      if (att) {
        // Convert degrees to radians and smoothly interpolate
        const targetRoll = THREE.MathUtils.degToRad(att.roll);
        const targetPitch = THREE.MathUtils.degToRad(att.pitch);
        const targetYaw = THREE.MathUtils.degToRad(att.yaw);
        arm1.rotation.z = THREE.MathUtils.lerp(arm1.rotation.z, targetRoll * 0.5 - 0.2, 0.1);
        arm2.rotation.x = THREE.MathUtils.lerp(arm2.rotation.x, targetPitch * 0.5, 0.1);
        arm3.rotation.y = THREE.MathUtils.lerp(arm3.rotation.y, targetYaw * 0.5, 0.1);
      } else {
        arm1.rotation.z = Math.sin(t * 0.5) * 0.3 - 0.2;
        arm2.rotation.z = Math.sin(t * 0.5 + 0.5) * 0.4;
        arm3.rotation.z = Math.sin(t * 0.5 + 1.0) * 0.5;
      }
      arm2.position.x = arm1.position.x + 0.2;
      arm3.position.x = arm2.position.x + 0.15;

      // Cyan light pulsing
      cyan_light.intensity = 1.2 + Math.sin(t * 2) * 0.3;

      controls.update();
      renderer.render(scene, camera);
    };
    animate();

    const handleResize = () => {
      const nw = mount.clientWidth;
      const nh = mount.clientHeight;
      camera.aspect = nw / nh;
      camera.updateProjectionMatrix();
      renderer.setSize(nw, nh);
    };
    window.addEventListener('resize', handleResize);

    return () => {
      cancelAnimationFrame(rafId);
      window.removeEventListener('resize', handleResize);
      controls.dispose();
      renderer.dispose();
      mount.removeChild(renderer.domElement);
    };
  }, []);

  return <div ref={mountRef} className={className} style={{ width: '100%', height: '100%' }} />;
};