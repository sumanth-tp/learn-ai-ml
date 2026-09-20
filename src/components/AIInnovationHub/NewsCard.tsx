import type {Discovery} from './types';
import styles from './styles.module.css';

const ICONS = {papers: '⌁', models: '◇', tools: '⌘', videos: '▶'};

function formatDate(value: string) {
  if (!value) return 'Recently published';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleDateString(undefined, {day: 'numeric', month: 'short', year: 'numeric'});
}

export default function NewsCard({item}: {item: Discovery}) {
  return (
    <article className={styles.card} data-category={item.category}>
      {item.image && <a href={item.url} target="_blank" rel="noreferrer" className={styles.thumb}><img src={item.image} alt="" loading="lazy" /></a>}
      <div className={styles.cardBody}>
        <div className={styles.cardMeta}>
          <span className={styles.categoryIcon} aria-hidden="true">{ICONS[item.category]}</span>
          <span>{item.source}</span><span>·</span><time dateTime={item.publishedAt}>{formatDate(item.publishedAt)}</time>
        </div>
        <h3><a href={item.url} target="_blank" rel="noreferrer">{item.title}</a></h3>
        {item.authors && item.authors.length > 0 && <p className={styles.authors}>{item.authors.join(', ')}</p>}
        <p className={styles.explanation}>{item.simpleExplanation}</p>
        <div className={styles.impact}>
          <div className={styles.why}><strong>Why it matters</strong><span>{item.whyImportant}</span></div>
          <div className={styles.useCases}>
            <strong>Useful for</strong>
            <ul>{item.useCases.map((useCase) => <li key={useCase}>{useCase}</li>)}</ul>
          </div>
        </div>
        {(item.prerequisites.length > 0 || item.projectIdeas.length > 0) && (
          <div className={styles.learningMeta}>
            {item.prerequisites.length > 0 && <div><b>Know first</b>{item.prerequisites.join(' · ')}</div>}
            {item.projectIdeas.length > 0 && <div><b>Try building</b>{item.projectIdeas.join(' · ')}</div>}
          </div>
        )}
        <footer className={styles.cardFoot}>
          <div className={styles.metrics}>{item.metrics?.map((metric) => <span key={metric}>{metric}</span>)}</div>
          <a href={item.url} target="_blank" rel="noreferrer">Open source <span aria-hidden="true">↗</span></a>
        </footer>
      </div>
    </article>
  );
}
