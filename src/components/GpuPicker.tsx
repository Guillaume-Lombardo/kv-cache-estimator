"use client";

import type { GpuSpec } from "@/lib/config-schema";
import { GPU_GROUPS, familyOf, gpuFamilies } from "@/lib/gpu-groups";
import { ChipGroup, type ChipOption } from "./ChipGroup";
import { usePrefs } from "./Prefs";

/** "80 GB HBM3 (…)" → "80 GB". */
const memShort = (g: GpuSpec) => g.memoryLabel.split(" (")[0].split(" ").slice(0, 2).join(" ");

/**
 * Two-step GPU choice: a family among a few labelled groups (NVIDIA data center, AMD data center,
 * PCIe cards), then a variant (memory size, form factor, rack-scale domain) when the family has several.
 */
export function GpuPicker({ gpus, value, onChange }: { gpus: GpuSpec[]; value: string; onChange: (id: string) => void }) {
  const { lang, t, fmt } = usePrefs();
  const families = gpuFamilies(gpus);
  const current = gpus.find((g) => g.id === value)!;
  const family = families.find((f) => f.variants.some((g) => g.id === value))!;

  const link = (g: GpuSpec) =>
    !g.interconnect.nvlink ? (
      <span className="no-link">PCIe</span>
    ) : g.nvlinkDomain > 8 ? (
      `${g.interconnect.fabric} ×${g.nvlinkDomain}`
    ) : (
      `${g.interconnect.fabric}${g.nvlinkDomain < g.maxGpusPerNode ? ` ×${g.nvlinkDomain}` : ""}`
    );

  const familyOption = (f: (typeof families)[number]): ChipOption => {
    const mems = [...new Set(f.variants.map(memShort))].sort((a, b) => parseFloat(a) - parseFloat(b));
    return {
      id: f.name,
      label: f.name,
      sub: (
        <>
          {mems.length > 1 ? `${mems[0].split(" ")[0]}–${mems[mems.length - 1]}` : mems[0]}
          {f.variants.length > 1 && <span className="muted"> · {t(`${f.variants.length} variantes`, `${f.variants.length} variants`)}</span>}
        </>
      ),
      title: f.variants.map((g) => `${g.name} · ${g.architecture} · ${g.memoryLabel} · ${fmt.num(g.bandwidthTBs)} ${t("To/s", "TB/s")}`).join("\n"),
    };
  };

  const variantOptions: ChipOption[] = family.variants.map((g) => ({
    id: g.id,
    label: g.variant ?? g.name,
    sub: (
      <>
        {fmt.num(g.bandwidthTBs)} {t("To/s", "TB/s")} · {link(g)}
      </>
    ),
    title: `${g.name} · ${g.architecture} · ${g.memoryLabel} · ${g.interconnect.label}`,
  }));

  return (
    <div className="gpu-picker">
      {GPU_GROUPS.map((group) => {
        const members = families.filter((f) => f.group === group.id);
        if (!members.length) return null;
        return (
          <div key={group.id} className="gpu-group">
            <span className="gpu-group-label">{group[lang]}</span>
            <ChipGroup
              label={group[lang]}
              options={members.map(familyOption)}
              value={familyOf(current)}
              onChange={(name) => onChange(families.find((f) => f.name === name && f.group === group.id)!.variants[0].id)}
              className="gpu-grid"
            />
          </div>
        );
      })}
      {family.variants.length > 1 && (
        <div className="gpu-variants">
          <span className="gpu-group-label">
            {t("Variante de", "Variant of")} {family.name}
          </span>
          <ChipGroup label={t(`Variante de ${family.name}`, `${family.name} variant`)} options={variantOptions} value={value} onChange={onChange} />
        </div>
      )}
      <p className="hint">
        <strong>{current.name}</strong> · {current.architecture} · {current.memoryLabel} · {fmt.num(current.bandwidthTBs)} {t("To/s", "TB/s")}
        {current.priceUsdPerHour ? t(` · ≈ ${current.priceUsdPerHour} $/h`, ` · ≈ $${current.priceUsdPerHour}/h`) : ""}
      </p>
    </div>
  );
}
