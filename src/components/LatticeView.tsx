/**
 * LatticeView: Three.js renderer for a 2D Ising spin configuration.
 *
 * One InstancedMesh of unit quads, one instance per lattice site, viewed through
 * an orthographic camera aimed straight down the +Z axis so the lattice reads as
 * a flat, crisp grid. Instance matrices are written once per lattice size; every
 * incoming frame touches only the per-instance colour buffer, which is what keeps
 * a 128x128 lattice (16384 instances) at 60fps.
 *
 * Colour choice: a warm amber for up spins against a cool cyan-blue for down
 * spins. Deliberately not red/green, so the image survives the common forms of
 * colour-vision deficiency, and both are bright enough to read on a dark page.
 */

import { useEffect, useRef } from 'react';
import * as THREE from 'three';
import type { Frame } from '../types';

export interface LatticeViewProps {
  frame: Frame | null;
  className?: string;
}

/** Warm: spin = +1. */
const COLOR_UP = new THREE.Color(0xff9e3d);
/** Cool: spin = -1. */
const COLOR_DOWN = new THREE.Color(0x3fa9f5);

/** Fraction of extra frustum around the lattice so the outermost row of sites is
 *  not clipped by the viewport edge. */
const VIEW_MARGIN = 1.04;

export function LatticeView(props: LatticeViewProps): JSX.Element {
  const { frame, className } = props;

  const containerRef = useRef<HTMLDivElement | null>(null);

  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.OrthographicCamera | null>(null);
  const geometryRef = useRef<THREE.PlaneGeometry | null>(null);
  const materialRef = useRef<THREE.MeshBasicMaterial | null>(null);
  const meshRef = useRef<THREE.InstancedMesh | null>(null);

  /** Lattice edge length currently baked into the InstancedMesh; 0 means none. */
  const latticeSizeRef = useRef<number>(0);
  const rafRef = useRef<number | null>(null);

  /* Stable handles into the mount effect's closures, so the per-frame effect can
   * re-fit the camera and request a draw without re-running setup. */
  const layoutRef = useRef<(() => void) | null>(null);
  const scheduleRef = useRef<(() => void) | null>(null);

  /* ---------------------------------------------------------------- */
  /* Mount: renderer, scene, camera, resize observer                   */
  /* ---------------------------------------------------------------- */

  useEffect(() => {
    const container = containerRef.current;
    if (container === null) return;

    const renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: true,
      powerPreference: 'high-performance',
    });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setClearColor(0x000000, 0);
    renderer.domElement.style.display = 'block';
    renderer.domElement.style.width = '100%';
    renderer.domElement.style.height = '100%';
    container.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, 100);
    camera.position.set(0, 0, 10);
    camera.lookAt(0, 0, 0);

    const geometry = new THREE.PlaneGeometry(1, 1);
    const material = new THREE.MeshBasicMaterial({
      color: 0xffffff,
      toneMapped: false,
      side: THREE.FrontSide,
    });

    rendererRef.current = renderer;
    sceneRef.current = scene;
    cameraRef.current = camera;
    geometryRef.current = geometry;
    materialRef.current = material;

    /** Fit the frustum to the container while keeping the lattice square,
     *  centred and entirely visible. */
    const layout = (): void => {
      const width = Math.max(1, container.clientWidth);
      const height = Math.max(1, container.clientHeight);
      renderer.setSize(width, height, false);

      // Half-extent of the lattice in world units (one unit per site).
      const half = Math.max(1, latticeSizeRef.current) * 0.5 * VIEW_MARGIN;
      const aspect = width / height;
      if (aspect >= 1) {
        camera.left = -half * aspect;
        camera.right = half * aspect;
        camera.top = half;
        camera.bottom = -half;
      } else {
        camera.left = -half;
        camera.right = half;
        camera.top = half / aspect;
        camera.bottom = -half / aspect;
      }
      camera.updateProjectionMatrix();
    };

    const draw = (): void => {
      rafRef.current = null;
      const r = rendererRef.current;
      const s = sceneRef.current;
      const c = cameraRef.current;
      if (r === null || s === null || c === null) return;
      r.render(s, c);
    };

    const schedule = (): void => {
      if (rafRef.current !== null) return;
      rafRef.current = requestAnimationFrame(draw);
    };

    layoutRef.current = layout;
    scheduleRef.current = schedule;

    layout();
    schedule();

    const observer = new ResizeObserver(() => {
      layout();
      schedule();
    });
    observer.observe(container);

    return () => {
      observer.disconnect();
      if (rafRef.current !== null) {
        cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }
      layoutRef.current = null;
      scheduleRef.current = null;

      const mesh = meshRef.current;
      if (mesh !== null) {
        scene.remove(mesh);
        mesh.dispose();
        meshRef.current = null;
      }
      latticeSizeRef.current = 0;

      scene.clear();
      geometry.dispose();
      material.dispose();
      renderer.dispose();
      renderer.forceContextLoss();
      if (renderer.domElement.parentNode !== null) {
        renderer.domElement.parentNode.removeChild(renderer.domElement);
      }

      rendererRef.current = null;
      sceneRef.current = null;
      cameraRef.current = null;
      geometryRef.current = null;
      materialRef.current = null;
    };
  }, []);

  /* ---------------------------------------------------------------- */
  /* Per-frame: colours only                                           */
  /* ---------------------------------------------------------------- */

  useEffect(() => {
    const scene = sceneRef.current;
    const geometry = geometryRef.current;
    const material = materialRef.current;
    const schedule = scheduleRef.current;
    const layout = layoutRef.current;
    if (scene === null || geometry === null || material === null) return;
    if (schedule === null || layout === null) return;

    // Empty state: tear the lattice down and show nothing.
    if (frame === null) {
      const existing = meshRef.current;
      if (existing !== null) {
        scene.remove(existing);
        existing.dispose();
        meshRef.current = null;
        latticeSizeRef.current = 0;
        layout();
      }
      schedule();
      return;
    }

    const size = frame.size;
    if (!Number.isFinite(size) || size <= 0) return;

    const count = size * size;

    // Rebuild only when the lattice size changes.
    if (meshRef.current === null || latticeSizeRef.current !== size) {
      const previous = meshRef.current;
      if (previous !== null) {
        scene.remove(previous);
        previous.dispose();
        meshRef.current = null;
      }

      const mesh = new THREE.InstancedMesh(geometry, material, count);
      mesh.frustumCulled = false;

      // Site (row, col) -> world position. Row 0 is drawn at the top.
      const offset = (size - 1) * 0.5;
      const matrix = new THREE.Matrix4();
      for (let row = 0; row < size; row++) {
        const y = offset - row;
        const base = row * size;
        for (let col = 0; col < size; col++) {
          matrix.makeTranslation(col - offset, y, 0);
          mesh.setMatrixAt(base + col, matrix);
        }
      }
      mesh.instanceMatrix.needsUpdate = true;
      mesh.instanceMatrix.setUsage(THREE.StaticDrawUsage);

      const colors = new THREE.InstancedBufferAttribute(new Float32Array(count * 3), 3);
      colors.setUsage(THREE.DynamicDrawUsage);
      mesh.instanceColor = colors;

      scene.add(mesh);
      meshRef.current = mesh;
      latticeSizeRef.current = size;
      layout();
    }

    const mesh = meshRef.current;
    const instanceColor = mesh.instanceColor;
    if (instanceColor === null) return;

    const spins = frame.spins;
    const target = instanceColor.array as Float32Array;
    const n = Math.min(count, spins.length);

    const ur = COLOR_UP.r;
    const ug = COLOR_UP.g;
    const ub = COLOR_UP.b;
    const dr = COLOR_DOWN.r;
    const dg = COLOR_DOWN.g;
    const db = COLOR_DOWN.b;

    for (let i = 0, j = 0; i < n; i++, j += 3) {
      if (spins[i] > 0) {
        target[j] = ur;
        target[j + 1] = ug;
        target[j + 2] = ub;
      } else {
        target[j] = dr;
        target[j + 1] = dg;
        target[j + 2] = db;
      }
    }
    instanceColor.needsUpdate = true;

    schedule();
  }, [frame]);

  /* ---------------------------------------------------------------- */

  return (
    <div
      ref={containerRef}
      className={className}
      style={{ position: 'relative', width: '100%', height: '100%', overflow: 'hidden' }}
    >
      {frame === null ? (
        <div
          style={{
            position: 'absolute',
            inset: 0,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            textAlign: 'center',
            padding: '1rem',
            font: '500 0.85rem/1.5 ui-sans-serif, system-ui, sans-serif',
            letterSpacing: '0.02em',
            color: 'rgba(190, 205, 220, 0.55)',
            pointerEvents: 'none',
          }}
        >
          No lattice yet — start the simulation to see the spins.
        </div>
      ) : null}
    </div>
  );
}
