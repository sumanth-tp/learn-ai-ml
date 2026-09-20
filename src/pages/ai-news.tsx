import Heading from '@theme/Heading';
import Layout from '@theme/Layout';
import type {ReactNode} from 'react';

import AIInnovationHub from '@site/src/components/AIInnovationHub';

import styles from './ai-news.module.css';

export default function AINewsPage(): ReactNode {
  return (
    <Layout title="AI Innovation Hub" description="Discover recent AI papers, models, open-source tools and videos on demand.">
      <main className={styles.page}>
        <div className="container">
          <header className={styles.hero}>
            <span className={styles.eyebrow}>AI Innovation Hub</span>
            <Heading as="h1">Fresh AI developments,<br /><span>when you ask for them.</span></Heading>
            <p>One click checks public research, model and developer sources. Gemini or Groq can turn the raw records into concise learning notes, while every claim stays linked to the original.</p>
            <div className={styles.sourceLine}><span>OpenAlex</span><i /> <span>Hugging Face</span><i /> <span>GitHub</span><i /> <span>YouTube via community index</span></div>
          </header>
          <AIInnovationHub />
          <footer className={styles.disclaimer}>Results are generated on demand and may be incomplete. Treat summaries as a reading shortlist, then verify details in the linked source.</footer>
        </div>
      </main>
    </Layout>
  );
}
