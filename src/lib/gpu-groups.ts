import type { GpuSpec } from "./config-schema";
import type { Lang } from "./i18n";

/** Picker groups: few enough to scan, each with cards of the same class. */
export const GPU_GROUPS = [
  { id: "nvidia-dc", fr: "GPU data center NVIDIA", en: "NVIDIA data center GPUs", match: (g: GpuSpec) => g.segment === "datacenter" && g.vendor === "nvidia" },
  { id: "amd-dc", fr: "GPU data center AMD", en: "AMD data center GPUs", match: (g: GpuSpec) => g.segment === "datacenter" && g.vendor === "amd" },
  { id: "pcie", fr: "GPU PCIe : inférence et stations", en: "PCIe GPUs: inference and workstations", match: (g: GpuSpec) => g.segment === "pcie" },
] as const;
export type GpuGroupId = (typeof GPU_GROUPS)[number]["id"];

export const gpuGroupOf = (g: GpuSpec): GpuGroupId => GPU_GROUPS.find((x) => x.match(g))!.id;
export const familyOf = (g: GpuSpec) => g.family ?? g.name;

export interface GpuFamily {
  name: string;
  group: GpuGroupId;
  /** Catalogue order. */
  variants: GpuSpec[];
}

/** Families in catalogue order, grouped for the picker. */
export function gpuFamilies(gpus: GpuSpec[]): GpuFamily[] {
  const out: GpuFamily[] = [];
  for (const g of gpus) {
    const name = familyOf(g);
    const group = gpuGroupOf(g);
    const f = out.find((x) => x.name === name && x.group === group);
    if (f) f.variants.push(g);
    else out.push({ name, group, variants: [g] });
  }
  return out;
}

export const groupLabel = (id: GpuGroupId, lang: Lang) => GPU_GROUPS.find((g) => g.id === id)![lang];
