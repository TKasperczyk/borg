type ExpandableTextProps = {
  text: string;
  expanded: boolean;
  className: string;
  expandedClassName: string;
};

export function ExpandableText({
  text,
  expanded,
  className,
  expandedClassName,
}: ExpandableTextProps) {
  return <p className={expanded ? `${className} ${expandedClassName}` : className}>{text}</p>;
}
