import React, { useRef, useEffect } from 'react';
import * as THREE from 'three';

interface SpinningGlobeProps {
  className?: string;
}

function buildEarthTexture(): THREE.CanvasTexture {
  const size = 512;
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d')!;

  // Ocean base
  ctx.fillStyle = '#0a1628';
  ctx.fillRect(0, 0, size, size);

  // Land masses (rough polygons)
  ctx.fillStyle = '#1a4a2a';
  const continents = [
    // North America
    [[50,60],[130,55],[145,90],[120,150],[80,170],[50,140],[30,100]],
    // South America
    [[100,180],[130,175],[135,260],[105,310],[85,285],[80,220]],
    // Europe/Africa
    [[200,55],[255,50],[260,80],[245,130],[255,200],[240,310],[210,320],[195,250],[185,180],[195,120],[190,80]],
    // Asia
    [[255,50],[380,45],[400,80],[390,150],[350,180],[310,160],[280,130],[255,120],[245,80]],
    // Australia
    [[340,230],[395,220],[405,270],[370,300],[330,280],[320,250]],
  ];
  continents.forEach(points => {
    ctx.beginPath();
    points.forEach(([x, y], i) => i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y));
    ctx.closePath();
    ctx.fill();
  });

  // Polar ice caps
  ctx.fillStyle = 'rgba(200,230,255,0.4)';
  ctx.beginPath();
  ctx.ellipse(size/2, 15, size/3, 20, 0, 0, Math.PI * 2);
  ctx.fill();
  ctx.beginPath();
  ctx.ellipse(size/2, size-15, size/4, 15, 0, 0, Math.PI * 2);
  ctx.fill();

  // Cloud wisps
  ctx.fillStyle = 'rgba(255,255,255,0.08)';
  for (let i = 0; i < 8; i++) {
    const x = Math.random() * size;
    const y = Math.random() * size;
    ctx.beginPath();
    ctx.ellipse(x, y, 60 + Math.random() * 40, 8 + Math.random() * 6, Math.random() * Math.PI, 0, Math.PI * 2);
    ctx.fill();
  }

  return new THREE.CanvasTexture(canvas);
}

export const SpinningGlobe: React.FC<SpinningGlobeProps> = ({ className }) => {
  const mountRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;

    const w = mount.clientWidth;
    const h = mount.clientHeight;

    // Renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(w, h);
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setClearColor(0x000000, 0);
    mount.appendChild(renderer.domElement);

    // Scene
    const scene = new THREE.Scene();

    // Camera
    const camera = new THREE.PerspectiveCamera(45, w / h, 0.1, 100);
    camera.position.set(0, 0, 4.5);

    // Lights
    scene.add(new THREE.AmbientLight(0x112244, 1.5));
    const sun = new THREE.DirectionalLight(0xffffff, 2);
    sun.position.set(5, 3, 5);
    scene.add(sun);

    // Globe
    const globeGeo = new THREE.SphereGeometry(1.5, 64, 64);
    const texture = buildEarthTexture();
    const globeMat = new THREE.MeshStandardMaterial({ map: texture, roughness: 0.9, metalness: 0.1 });
    const globe = new THREE.Mesh(globeGeo, globeMat);
    globe.rotation.z = THREE.MathUtils.degToRad(23.4);
    scene.add(globe);

    // Atmosphere glow
    const atmGeo = new THREE.SphereGeometry(1.62, 32, 32);
    const atmMat = new THREE.MeshPhongMaterial({
      color: 0x38bdf8,
      transparent: true,
      opacity: 0.08,
      side: THREE.FrontSide,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });
    scene.add(new THREE.Mesh(atmGeo, atmMat));

    // Outer glow ring
    const outerGeo = new THREE.SphereGeometry(1.72, 16, 16);
    const outerMat = new THREE.MeshPhongMaterial({
      color: 0x0ea5e9,
      transparent: true,
      opacity: 0.03,
      side: THREE.BackSide,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });
    scene.add(new THREE.Mesh(outerGeo, outerMat));

    // Lat/lon grid lines
    const gridMat = new THREE.LineBasicMaterial({ color: 0x38bdf8, transparent: true, opacity: 0.1 });
    for (let lat = -60; lat <= 60; lat += 30) {
      const points: THREE.Vector3[] = [];
      for (let lon = 0; lon <= 360; lon += 5) {
        const phi = THREE.MathUtils.degToRad(90 - lat);
        const theta = THREE.MathUtils.degToRad(lon);
        points.push(new THREE.Vector3().setFromSphericalCoords(1.52, phi, theta));
      }
      scene.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(points), gridMat));
    }

    // Animation loop
    let rafId: number;
    const animate = () => {
      rafId = requestAnimationFrame(animate);
      globe.rotation.y += 0.001;
      renderer.render(scene, camera);
    };
    animate();

    // Resize
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
      mount.removeChild(renderer.domElement);
      renderer.dispose();
      globeGeo.dispose();
      globeMat.dispose();
      texture.dispose();
    };
  }, []);

  return <div ref={mountRef} className={className} style={{ width: '100%', height: '100%' }} />;
};

export default SpinningGlobe;