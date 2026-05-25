export type SparkProps = {
  data: readonly number[];
  max?: number;
};

export function Spark({ data, max }: SparkProps) {
  const computedMax = max ?? Math.max(...data, 1);
  return (
    <div className="spark">
      {data.map((value, index) => (
        <div key={index} className="bar" style={{ height: `${(value / computedMax) * 100}%` }}></div>
      ))}
    </div>
  );
}
