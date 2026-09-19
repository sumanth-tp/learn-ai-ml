import Mermaid from '@theme-original/Mermaid';
import type MermaidType from '@theme/Mermaid';
import type {WrapperProps} from '@docusaurus/types';
import {JSX, useCallback, useEffect, useRef, useState} from 'react';
import {createPortal} from 'react-dom';

import styles from './styles.module.css';

type Props = WrapperProps<typeof MermaidType>;

const MIN_ZOOM = 0.25;
const MAX_ZOOM = 8;
const ZOOM_STEP = 1.25;

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

type Point = {x: number; y: number};

function Lightbox({svg, onClose}: {svg: string; onClose: () => void}) {
  const [zoom, setZoom] = useState(1);
  const [offset, setOffset] = useState<Point>({x: 0, y: 0});
  const dragOrigin = useRef<{pointer: Point; offset: Point} | null>(null);
  const stageRef = useRef<HTMLDivElement>(null);

  const reset = useCallback(() => {
    setZoom(1);
    setOffset({x: 0, y: 0});
  }, []);

  const zoomBy = useCallback((factor: number) => {
    setZoom((current) => clamp(current * factor, MIN_ZOOM, MAX_ZOOM));
  }, []);

  // Escape to close, +/- to zoom, 0 to reset.
  useEffect(() => {
    function onKeyDown(event: KeyboardEvent) {
      if (event.key === 'Escape') {
        onClose();
      } else if (event.key === '+' || event.key === '=') {
        zoomBy(ZOOM_STEP);
      } else if (event.key === '-' || event.key === '_') {
        zoomBy(1 / ZOOM_STEP);
      } else if (event.key === '0') {
        reset();
      }
    }
    document.addEventListener('keydown', onKeyDown);
    return () => document.removeEventListener('keydown', onKeyDown);
  }, [onClose, reset, zoomBy]);

  // Keep the page behind the overlay still while it is open.
  useEffect(() => {
    const previous = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    return () => {
      document.body.style.overflow = previous;
    };
  }, []);

  // Wheel zoom has to be bound manually: React marks wheel listeners passive,
  // so preventDefault there would be ignored and the page would scroll too.
  useEffect(() => {
    const stage = stageRef.current;
    if (!stage) return undefined;

    function onWheel(event: WheelEvent) {
      event.preventDefault();
      setZoom((current) =>
        clamp(current * (event.deltaY < 0 ? ZOOM_STEP : 1 / ZOOM_STEP), MIN_ZOOM, MAX_ZOOM),
      );
    }

    stage.addEventListener('wheel', onWheel, {passive: false});
    return () => stage.removeEventListener('wheel', onWheel);
  }, []);

  function onPointerDown(event: React.PointerEvent<HTMLDivElement>) {
    dragOrigin.current = {
      pointer: {x: event.clientX, y: event.clientY},
      offset,
    };
    event.currentTarget.setPointerCapture(event.pointerId);
  }

  function onPointerMove(event: React.PointerEvent<HTMLDivElement>) {
    const origin = dragOrigin.current;
    if (!origin) return;
    setOffset({
      x: origin.offset.x + (event.clientX - origin.pointer.x),
      y: origin.offset.y + (event.clientY - origin.pointer.y),
    });
  }

  function endDrag(event: React.PointerEvent<HTMLDivElement>) {
    dragOrigin.current = null;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
  }

  return createPortal(
    <div
      className={styles.overlay}
      role="dialog"
      aria-modal="true"
      aria-label="Enlarged diagram"
      onClick={onClose}>
      <div className={styles.toolbar} onClick={(event) => event.stopPropagation()}>
        <button
          type="button"
          className={styles.toolButton}
          onClick={() => zoomBy(1 / ZOOM_STEP)}
          disabled={zoom <= MIN_ZOOM}
          aria-label="Zoom out">
          &minus;
        </button>
        <span className={styles.zoomLabel}>{Math.round(zoom * 100)}%</span>
        <button
          type="button"
          className={styles.toolButton}
          onClick={() => zoomBy(ZOOM_STEP)}
          disabled={zoom >= MAX_ZOOM}
          aria-label="Zoom in">
          +
        </button>
        <button type="button" className={styles.toolButton} onClick={reset} aria-label="Reset zoom">
          Reset
        </button>
        <button
          type="button"
          className={`${styles.toolButton} ${styles.closeButton}`}
          onClick={onClose}
          aria-label="Close enlarged diagram">
          Close
        </button>
      </div>

      <div
        ref={stageRef}
        className={styles.stage}
        onClick={(event) => event.stopPropagation()}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={endDrag}
        onPointerCancel={endDrag}>
        <div
          className={styles.canvas}
          style={{
            transform: `translate(${offset.x}px, ${offset.y}px) scale(${zoom})`,
          }}
          // The SVG is produced by Mermaid from the page's own source.
          dangerouslySetInnerHTML={{__html: svg}}
        />
      </div>

      <p className={styles.hint}>Scroll to zoom · drag to pan · Esc to close</p>
    </div>,
    document.body,
  );
}

export default function MermaidWrapper(props: Props): JSX.Element {
  const containerRef = useRef<HTMLDivElement>(null);
  const [svg, setSvg] = useState<string | null>(null);

  function expand() {
    const rendered = containerRef.current?.querySelector('svg');
    if (!rendered) return;

    const clone = rendered.cloneNode(true) as SVGElement;
    // Mermaid constrains the inline diagram to the column width; the overlay
    // should size to its own viewport instead.
    clone.removeAttribute('width');
    clone.removeAttribute('height');
    clone.style.maxWidth = 'none';
    clone.style.width = '100%';
    clone.style.height = 'auto';

    setSvg(clone.outerHTML);
  }

  return (
    <div className={styles.wrapper} ref={containerRef}>
      <Mermaid {...props} />
      <button
        type="button"
        className={styles.expandButton}
        onClick={expand}
        aria-label="View diagram larger"
        title="View larger">
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" aria-hidden="true">
          <path
            d="M9 3H3v6M15 3h6v6M9 21H3v-6M15 21h6v-6"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
        <span>Expand</span>
      </button>

      {svg !== null && <Lightbox svg={svg} onClose={() => setSvg(null)} />}
    </div>
  );
}
