import {useColorMode} from '@docusaurus/theme-common';
import useIsBrowser from '@docusaurus/useIsBrowser';
import type {ReactNode} from 'react';
import {useState} from 'react';

import styles from './VizPanel.module.css';

export type LegendItem = {label: string; color: string; note?: string};

type Props = {
  title: string;
  /** One sentence on what the reader should do or notice. */
  hint?: string;
  controls?: ReactNode;
  legend?: LegendItem[];
  /** Rows shown by the "table" toggle — identity is never colour-alone. */
  table?: {columns: string[]; rows: (string | number)[][]};
  children: ReactNode;
};

/**
 * Shared frame for every interactive figure: title, controls, plot, legend and
 * an always-available table view. Keeps the charts consistent and accessible
 * without repeating chrome in each one.
 */
export default function VizPanel({title, hint, controls, legend, table, children}: Props) {
  const [showTable, setShowTable] = useState(false);

  return (
    <figure className={styles.panel}>
      <figcaption className={styles.head}>
        <span className={styles.badge}>Interactive</span>
        <span className={styles.title}>{title}</span>
        {table && (
          <button
            type="button"
            className={styles.tableToggle}
            onClick={() => setShowTable((v) => !v)}
            aria-pressed={showTable}>
            {showTable ? 'show chart' : 'show data'}
          </button>
        )}
      </figcaption>

      {controls && <div className={styles.controls}>{controls}</div>}

      <div className={styles.plot}>
        {showTable && table ? (
          <div className={styles.tableWrap}>
            <table className={styles.table}>
              <thead>
                <tr>
                  {table.columns.map((c) => (
                    <th key={c}>{c}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {table.rows.map((row, i) => (
                  <tr key={i}>
                    {row.map((cell, j) => (
                      <td key={j}>{cell}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          children
        )}
      </div>

      {legend && legend.length > 0 && (
        <ul className={styles.legend}>
          {legend.map((item) => (
            <li key={item.label} className={styles.legendItem}>
              <span className={styles.swatch} style={{background: item.color}} aria-hidden="true" />
              {item.label}
              {item.note && <span className={styles.legendNote}>{item.note}</span>}
            </li>
          ))}
        </ul>
      )}

      {hint && <p className={styles.hint}>{hint}</p>}
    </figure>
  );
}

/** True when the site is in dark mode — charts pick their own dark steps. */
export function useDarkViz(): boolean {
  const isBrowser = useIsBrowser();
  const {colorMode} = useColorMode();
  return isBrowser && colorMode === 'dark';
}

export {styles as vizStyles};
