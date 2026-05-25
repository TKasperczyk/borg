export type PlaceholderScreenProps = {
  name: string;
};

export function PlaceholderScreen({ name }: PlaceholderScreenProps) {
  return (
    <div className="full-page">
      <div className="page-head">
        <h1>{name}</h1>
        <span className="desc">wiring in v2</span>
      </div>
      <div className="page-body" style={{ display: "grid", placeItems: "center" }}>
        <div className="notice">screen :: {name} (wiring in v2)</div>
      </div>
    </div>
  );
}
