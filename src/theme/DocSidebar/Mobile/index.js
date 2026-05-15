/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
import React, {useCallback, useEffect, useRef, useState} from 'react';
import clsx from 'clsx';
import {
  NavbarSecondaryMenuFiller,
  ThemeClassNames,
} from '@docusaurus/theme-common';
import {useNavbarMobileSidebar} from '@docusaurus/theme-common/internal';
import DocSidebarItems from '@theme/DocSidebarItems';
import styles from './styles.module.css';

const expandedCategorySelector = [
  'button.menu__caret[aria-expanded="true"]',
  'a[role="button"][aria-expanded="true"]',
].join(',');

// eslint-disable-next-line react/function-component-definition
const DocSidebarMobileSecondaryMenu = ({sidebar, path}) => {
  const mobileSidebar = useNavbarMobileSidebar();
  const menuRef = useRef(null);
  const [expandedCategoryCount, setExpandedCategoryCount] = useState(0);

  const getExpandedCategoryItems = useCallback(
    () => menuRef.current?.querySelectorAll(expandedCategorySelector) ?? [],
    [],
  );

  const updateExpandedCategoryCount = useCallback(() => {
    setExpandedCategoryCount(getExpandedCategoryItems().length);
  }, [getExpandedCategoryItems]);

  useEffect(() => {
    updateExpandedCategoryCount();

    const observer = new MutationObserver(updateExpandedCategoryCount);
    if (menuRef.current) {
      observer.observe(menuRef.current, {
        attributes: true,
        attributeFilter: ['aria-expanded'],
        childList: true,
        subtree: true,
      });
    }

    return () => observer.disconnect();
  }, [path, sidebar, updateExpandedCategoryCount]);

  const collapseAll = () => {
    const expandedItems = getExpandedCategoryItems();

    expandedItems?.forEach((item) => item.click());
    setExpandedCategoryCount(0);
  };

  return (
    <div ref={menuRef}>
      {expandedCategoryCount > 1 && (
        <div className={styles.sidebarActions}>
          <button
            type="button"
            className={clsx('clean-btn', styles.collapseAllButton)}
            onClick={collapseAll}>
            Collapse all
          </button>
        </div>
      )}
      <ul className={clsx(ThemeClassNames.docs.docSidebarMenu, 'menu__list')}>
        <DocSidebarItems
          items={sidebar}
          activePath={path}
          onItemClick={(item) => {
            // Mobile sidebar should only be closed if the category has a link
            if (item.type === 'category' && item.href) {
              mobileSidebar.toggle();
            }
            if (item.type === 'link') {
              mobileSidebar.toggle();
            }
          }}
          level={1}
        />
      </ul>
    </div>
  );
};

function DocSidebarMobile(props) {
  return (
    <NavbarSecondaryMenuFiller
      component={DocSidebarMobileSecondaryMenu}
      props={props}
    />
  );
}

export default React.memo(DocSidebarMobile);
