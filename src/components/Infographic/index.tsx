import useBaseUrl from '@docusaurus/useBaseUrl';
import {useState} from 'react';

import {Lightbox} from '@site/src/theme/Mermaid';
import mermaidStyles from '@site/src/theme/Mermaid/styles.module.css';

import styles from './styles.module.css';

type Props = {
  /** Path under /static, e.g. "/img/ai-security/mosaic.svg". */
  src: string;
  /** What the image shows, for screen readers and search. */
  alt: string;
  /** Short caption shown under the image. */
  caption?: string;
};

/**
 * A course infographic: a static board-style image with the same Expand,
 * zoom and pan lightbox that Mermaid diagrams get.
 */
export default function Infographic({src, alt, caption}: Props) {
  const url = useBaseUrl(src);
  const [open, setOpen] = useState(false);
  // draggable=false and pointer-events:none keep the browser's own image drag
  // from fighting the lightbox's pan gesture.
  const markup = `<img src="${url}" alt="" draggable="false" style="display:block;width:100%;height:auto;pointer-events:none;user-select:none" />`;

  return (
    <figure className={styles.figure}>
      <div className={mermaidStyles.wrapper}>
        <button type="button" className={styles.imageButton} onClick={() => setOpen(true)} aria-label={`Enlarge: ${alt}`}>
          <img src={url} alt={alt} loading="lazy" className={styles.image} />
        </button>
        <button
          type="button"
          className={mermaidStyles.expandButton}
          onClick={() => setOpen(true)}
          aria-label="View infographic larger"
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
      </div>
      {caption && <figcaption className={styles.caption}>{caption}</figcaption>}
      {open && <Lightbox svg={markup} onClose={() => setOpen(false)} />}
    </figure>
  );
}
