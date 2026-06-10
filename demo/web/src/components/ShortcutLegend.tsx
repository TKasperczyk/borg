import { RAIL_ITEMS, routeChordLabel } from "../routes";
import { Modal } from "./Modal";

export function ShortcutLegend({ open, onClose }: { open: boolean; onClose: () => void }) {
  return (
    <Modal open={open} title="shortcuts" onClose={onClose}>
      <div className="shortcut-legend">
        <section className="shortcut-section" aria-label="routes">
          <div className="shortcut-section-title">routes</div>
          <div className="shortcut-list">
            {RAIL_ITEMS.map((item) => (
              <div className="shortcut-row" key={item.id}>
                <span className="shortcut-route">
                  <span className="shortcut-glyph" aria-hidden="true">
                    {item.glyph}
                  </span>
                  <span>{item.title ?? item.label}</span>
                </span>
                <span className="kbd">{routeChordLabel(item)}</span>
              </div>
            ))}
          </div>
        </section>
        <section className="shortcut-section" aria-label="global shortcuts">
          <div className="shortcut-section-title">global</div>
          <div className="shortcut-list">
            <div className="shortcut-row">
              <span>command palette</span>
              <span className="kbd">ctrl+K</span>
            </div>
            <div className="shortcut-row">
              <span>shortcut legend</span>
              <span className="kbd">?</span>
            </div>
            <div className="shortcut-row">
              <span>close overlays</span>
              <span className="kbd">esc</span>
            </div>
          </div>
        </section>
      </div>
    </Modal>
  );
}
