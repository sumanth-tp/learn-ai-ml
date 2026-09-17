import Heading from '@theme/Heading';
import type {ComponentType, ReactNode} from 'react';

import {BoltIcon, BrainIcon, ChartIcon} from '@site/src/components/Icons';

import styles from './styles.module.css';

type FeatureItem = {
  title: string;
  Icon: ComponentType<{className?: string}>;
  description: ReactNode;
};

const FEATURES: FeatureItem[] = [
  {
    title: 'Derived, not memorised',
    Icon: BrainIcon,
    description: (
      <>
        Every result is built up from the maths behind it — gradients, losses and
        distributions are worked through before any library is imported.
      </>
    ),
  },
  {
    title: 'Code you can run',
    Icon: BoltIcon,
    description: (
      <>
        NumPy, pandas, scikit-learn, PyTorch and FastAPI examples are complete and
        copy-pasteable, with the failure modes called out beside them.
      </>
    ),
  },
  {
    title: 'Visual by default',
    Icon: ChartIcon,
    description: (
      <>
        Hundreds of generated plots and diagrams explain distributions,
        convergence and architectures far faster than prose can.
      </>
    ),
  },
];

export default function HomepageFeatures(): ReactNode {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className={styles.grid}>
          {FEATURES.map(({title, Icon, description}) => (
            <div key={title} className={styles.feature}>
              <span className={styles.iconWrap}>
                <Icon className={styles.icon} />
              </span>
              <Heading as="h3" className={styles.title}>
                {title}
              </Heading>
              <p className={styles.description}>{description}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
