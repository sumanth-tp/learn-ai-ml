import {useMemo, useState} from 'react';

import {seriesColor} from './palette';
import VizPanel, {useDarkViz, vizStyles as s} from './VizPanel';

export const PLATFORMS = ["SageMaker AI", "Vertex (Agent Platform)", "Azure ML", "Bedrock", "Foundry"];
export const CAPABILITIES: {key: string; label: string; cells: Record<string, string>}[] = [
  {
    "key": "custom_training",
    "label": "Train your own models",
    "cells": {
      "SageMaker AI": "sm_whatis",
      "Vertex (Agent Platform)": "vtx_intro",
      "Azure ML": "azml_overview"
    }
  },
  {
    "key": "pipelines",
    "label": "Managed ML pipelines",
    "cells": {
      "SageMaker AI": "sm_features",
      "Vertex (Agent Platform)": "vtx_pipe",
      "Azure ML": "azml_overview"
    }
  },
  {
    "key": "model_registry",
    "label": "Model registry",
    "cells": {
      "SageMaker AI": "sm_features",
      "Vertex (Agent Platform)": "vtx_intro",
      "Azure ML": "azml_mlops"
    }
  },
  {
    "key": "feature_store",
    "label": "Feature store",
    "cells": {
      "SageMaker AI": "sm_features",
      "Vertex (Agent Platform)": "vtx_intro"
    }
  },
  {
    "key": "experiment_tracking",
    "label": "Experiment tracking",
    "cells": {
      "SageMaker AI": "sm_features",
      "Vertex (Agent Platform)": "vtx_intro",
      "Azure ML": "azml_overview"
    }
  },
  {
    "key": "drift_monitoring",
    "label": "Production drift monitoring",
    "cells": {
      "SageMaker AI": "sm_features",
      "Vertex (Agent Platform)": "vtx_monitor",
      "Azure ML": "azml_mlops"
    }
  },
  {
    "key": "online_endpoint",
    "label": "Online inference endpoint",
    "cells": {
      "SageMaker AI": "sm_deploy",
      "Vertex (Agent Platform)": "vtx_intro",
      "Azure ML": "azml_endpoints",
      "Bedrock": "bed_overview",
      "Foundry": "foundry"
    }
  },
  {
    "key": "batch_inference",
    "label": "Batch inference",
    "cells": {
      "SageMaker AI": "sm_batch",
      "Vertex (Agent Platform)": "vtx_batch",
      "Azure ML": "azml_endpoints",
      "Bedrock": "bed_batch"
    }
  },
  {
    "key": "serverless_inference",
    "label": "Serverless inference",
    "cells": {
      "SageMaker AI": "sm_deploy",
      "Azure ML": "azml_endpoints"
    }
  },
  {
    "key": "foundation_models",
    "label": "Hosted foundation models",
    "cells": {
      "SageMaker AI": "sm_deploy",
      "Vertex (Agent Platform)": "vtx_intro",
      "Azure ML": "azml_overview",
      "Bedrock": "bed_overview",
      "Foundry": "foundry"
    }
  },
  {
    "key": "managed_rag",
    "label": "Managed retrieval (RAG)",
    "cells": {
      "Bedrock": "bed_kb"
    }
  },
  {
    "key": "guardrails",
    "label": "Guardrails or content filters",
    "cells": {
      "Bedrock": "bed_guard",
      "Foundry": "foundry"
    }
  },
  {
    "key": "agents",
    "label": "Agent building",
    "cells": {
      "Vertex (Agent Platform)": "vtx_intro",
      "Bedrock": "bed_agents",
      "Foundry": "foundry"
    }
  },
  {
    "key": "kubernetes_compute",
    "label": "Kubernetes compute",
    "cells": {
      "SageMaker AI": "sm_features",
      "Azure ML": "azml_overview"
    }
  }
];
export const SCENARIOS: Record<string, string[]> = {
  "classical ML team": [
    "custom_training",
    "pipelines",
    "model_registry",
    "experiment_tracking",
    "drift_monitoring",
    "online_endpoint",
    "batch_inference"
  ],
  "GenAI application team": [
    "foundation_models",
    "managed_rag",
    "guardrails",
    "agents",
    "batch_inference",
    "online_endpoint"
  ],
  "fine-tune and serve": [
    "custom_training",
    "foundation_models",
    "online_endpoint",
    "serverless_inference"
  ]
};

const W = 640;
const ROW = 78;

function wrapWords(text: string, width: number): string[] {
  const lines: string[] = [];
  let current = '';
  for (const word of text.split(' ')) {
    if ((current + ' ' + word).trim().length > width) {
      lines.push(current);
      current = word;
    } else {
      current = (current + ' ' + word).trim();
    }
  }
  if (current) lines.push(current);
  return lines;
}

export function coverage(required: string[]) {
  return PLATFORMS.map((platform) => {
    const have = required.filter((k) => CAPABILITIES.find((c) => c.key === k)!.cells[platform]);
    const missing = required
      .filter((k) => !CAPABILITIES.find((c) => c.key === k)!.cells[platform])
      .map((k) => CAPABILITIES.find((c) => c.key === k)!.label);
    return {platform, count: have.length, missing};
  }).sort((a, b) => b.count - a.count || a.platform.localeCompare(b.platform));
}

export default function PlatformChooserLab() {
  const dark = useDarkViz();
  const [scenario, setScenario] = useState('classical ML team');
  const [custom, setCustom] = useState<string[]>(SCENARIOS['classical ML team']);

  const required = scenario === 'custom' ? custom : SCENARIOS[scenario];
  const rows = useMemo(() => coverage(required), [required]);
  const height = 20 + rows.length * ROW + 10;

  const toggle = (key: string) => {
    const base = scenario === 'custom' ? custom : SCENARIOS[scenario];
    const next = base.includes(key) ? base.filter((k) => k !== key) : [...base, key];
    setCustom(next);
    setScenario('custom');
  };

  const bar = seriesColor(0, dark);
  const track = dark ? '#2b3340' : '#e3e6ea';
  const status = rows.map((r) => `${r.platform} ${r.count} of ${required.length}`).join(', ');

  return (
    <VizPanel
      title="Which platform pages show the capabilities you need"
      hint="Pick a scenario or tick your own list. A bar counts the required capabilities that an official documentation page I read shows for that platform. A missing capability means 'not found on the pages read', not 'not offered', so use the result to decide what to verify, not what to buy. Defaults match block 2: for the classical ML team three platforms tie at 7 of 7."
      table={{
        columns: ['capability', ...PLATFORMS],
        rows: CAPABILITIES.map((c) => [c.label, ...PLATFORMS.map((p) => c.cells[p] ?? 'not found')]),
      }}
      controls={
        <>
          <label className={s.control}>
            scenario
            <select className={s.select} value={scenario} onChange={(e) => setScenario(e.target.value)}>
              {Object.keys(SCENARIOS).map((name) => (
                <option key={name} value={name}>{name}</option>
              ))}
              <option value="custom">custom</option>
            </select>
          </label>
          <fieldset style={{border: 0, padding: 0, margin: 0, display: 'flex', flexWrap: 'wrap', gap: '0.25rem 1rem'}}>
            <legend className={s.value}>required capabilities</legend>
            {CAPABILITIES.map((c) => (
              <label key={c.key} className={s.control}>
                <input type="checkbox" checked={required.includes(c.key)} onChange={() => toggle(c.key)} /> {c.label}
              </label>
            ))}
          </fieldset>
          <span className={s.value} aria-live="polite">
            {required.length === 0 ? 'Select at least one capability.' : status}
          </span>
        </>
      }>
      <svg className={s.svg} viewBox={`0 0 ${W} ${height}`} role="img" aria-label={`Coverage of ${required.length} required capabilities: ${status}`}>
        {rows.map((r, i) => {
          const y = 20 + i * ROW;
          const full = required.length > 0 ? (r.count / required.length) * 360 : 0;
          return (
            <g key={r.platform}>
              <text className={s.dataLabel} x={10} y={y + 14}>
                {r.platform}
              </text>
              <rect x={200} y={y} width={360} height={18} rx={4} fill={track} />
              <rect x={200} y={y} width={full} height={18} rx={4} fill={bar} />
              <text className={s.dataLabel} x={570} y={y + 14}>
                {r.count} of {required.length}
              </text>
              {wrapWords(r.missing.length > 0 ? `not found: ${r.missing.join(', ')}` : '', 78).map((l, k) => (
                <text key={k} className={s.tick} x={200} y={y + 34 + k * 14}>
                  {l}
                </text>
              ))}
            </g>
          );
        })}
      </svg>
    </VizPanel>
  );
}
