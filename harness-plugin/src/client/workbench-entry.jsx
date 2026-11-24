import React from 'react';
import { ControlIcon } from './panel.jsx';
import './workbench-entry.css';

export function WorkbenchEntry({wide, panelId, usePanelInfo, openWorkbench}) {
  const active = usePanelInfo(info => info.activePanelId === panelId);
  return <button type="button" className="control-workbench-entry" data-compact={!wide || undefined}
    aria-label="控制工程工作台" aria-current={active ? 'page' : undefined}
    title={wide ? undefined : '控制工程工作台 · 天津大学 × 上海联合汽车电子有限公司'} onClick={openWorkbench}>
    <span className="control-entry-glyph" aria-hidden="true"><ControlIcon size={20}/></span>
    {wide && <span className="control-entry-copy">
      <strong>控制工程工作台</strong>
      <span>天津大学</span>
      <span>上海联合汽车电子有限公司</span>
    </span>}
  </button>;
}
