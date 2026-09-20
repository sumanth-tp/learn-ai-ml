import useBaseUrl from '@docusaurus/useBaseUrl';
import styles from './styles.module.css';

export default function PaperPdf({slug, title}: {slug: string; title: string}) {
  const src = useBaseUrl(`/papers/research-papers/${slug}.pdf`);
  return (
    <section className={styles.paper} aria-label={`Original paper: ${title}`}>
      <p><a href={src} target="_blank" rel="noreferrer">Open the complete paper</a>{' · '}<a href={src} download>Download PDF</a></p>
      <iframe src={`${src}#view=FitH`} title={`${title} — original PDF`} loading="lazy" className={styles.viewer} />
      <p className={styles.help}>If your browser cannot display the PDF here, open or download it using the links above.</p>
    </section>
  );
}
