import type {ReactNode} from 'react';

import SiteChrome from '@site/src/components/SiteChrome';

/**
 * Docusaurus renders <Root> once, above the router-driven page tree — the right
 * place for chrome that must survive navigation (reading progress, shortcuts).
 */
export default function Root({children}: {children: ReactNode}): ReactNode {
  return (
    <>
      <SiteChrome />
      {children}
    </>
  );
}
