import type { Plane } from "../api";
import type { OverlayMode } from "../lib/render";
import { Viewport } from "./Viewport";

export type LayoutMode = "single" | "stack" | "grid";

const PLANES: { plane: Plane; label: string }[] = [
  { plane: "axial", label: "Axial" },
  { plane: "coronal", label: "Coronal" },
  { plane: "sagittal", label: "Sagittal" },
];

interface ViewportGridProps {
  layout: LayoutMode;
  expandedPlane: Plane | null;
  onToggleExpand: (plane: Plane) => void;
  singlePlane: Plane;
  onChangeSinglePlane: (plane: Plane) => void;
  onFocusPlane: (plane: Plane) => void;
  sliceIndices: Record<Plane, number>;
  planeCounts: Record<Plane, number>;
  shape: [number, number, number] | null;
  image: Uint8Array | null;
  predictionMask: Uint8Array | null;
  labelMask: Uint8Array | null;
  uncertainty: Uint8Array | null;
  overlayMode: OverlayMode;
  overlayOpacity: number;
  showTruthOutline: boolean;
  showUncertainty: boolean;
  uncertaintyOpacity: number;
}

export function ViewportGrid(props: ViewportGridProps) {
  const {
    layout,
    expandedPlane,
    onToggleExpand,
    singlePlane,
    onChangeSinglePlane,
    onFocusPlane,
    sliceIndices,
    planeCounts,
    shape,
    image,
    predictionMask,
    labelMask,
    uncertainty,
    overlayMode,
    overlayOpacity,
    showTruthOutline,
    showUncertainty,
    uncertaintyOpacity,
  } = props;

  const shared = {
    shape,
    image,
    predictionMask,
    labelMask,
    uncertainty,
    overlayMode,
    overlayOpacity,
    showTruthOutline,
    showUncertainty,
    uncertaintyOpacity,
  };

  if (layout === "single") {
    return (
      <div className="flex h-full min-h-0 flex-col gap-2">
        <div className="flex shrink-0 gap-1">
          {PLANES.map(({ plane, label }) => (
            <button
              key={plane}
              type="button"
              onClick={() => onChangeSinglePlane(plane)}
              className={`rounded-sm border px-2 py-1 font-condensed text-[11px] tracking-[0.1em] uppercase transition-colors duration-[120ms] ${
                singlePlane === plane
                  ? "border-text-primary text-text-primary"
                  : "border-surface-seam text-text-secondary hover:text-text-primary"
              }`}
            >
              {label}
            </button>
          ))}
        </div>
        <div className="min-h-0 flex-1">
          <Viewport
            plane={singlePlane}
            planeLabel={PLANES.find((p) => p.plane === singlePlane)!.label}
            sliceIndex={sliceIndices[singlePlane]}
            sliceCount={planeCounts[singlePlane]}
            expanded
            expandable={false}
            onToggleExpand={() => {}}
            onFocusPlane={() => onFocusPlane(singlePlane)}
            {...shared}
          />
        </div>
      </div>
    );
  }

  if (expandedPlane) {
    const meta = PLANES.find((p) => p.plane === expandedPlane)!;
    return (
      <div className="h-full min-h-0">
        <Viewport
          plane={expandedPlane}
          planeLabel={meta.label}
          sliceIndex={sliceIndices[expandedPlane]}
          sliceCount={planeCounts[expandedPlane]}
          expanded
          expandable
          onToggleExpand={() => onToggleExpand(expandedPlane)}
          onFocusPlane={() => onFocusPlane(expandedPlane)}
          {...shared}
        />
      </div>
    );
  }

  const wrapperClass =
    layout === "grid"
      ? "grid h-full min-h-0 grid-cols-3 gap-2"
      : "flex h-full min-h-0 flex-col gap-2";

  return (
    <div className={wrapperClass}>
      {PLANES.map(({ plane, label }) => (
        <div key={plane} className="min-h-0" style={layout === "stack" ? { flex: 1 } : undefined}>
          <Viewport
            plane={plane}
            planeLabel={label}
            sliceIndex={sliceIndices[plane]}
            sliceCount={planeCounts[plane]}
            expanded={false}
            expandable
            onToggleExpand={() => onToggleExpand(plane)}
            onFocusPlane={() => onFocusPlane(plane)}
            {...shared}
          />
        </div>
      ))}
    </div>
  );
}
