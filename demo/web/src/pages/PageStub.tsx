type PageStubProps = {
  title: string;
  subtitle: string;
};

export function PageStub({ title, subtitle }: PageStubProps) {
  return (
    <main className="page">
      <header className="page-header">
        <span className="page-title">{title}</span>
        <span className="page-subtitle">{subtitle}</span>
      </header>
      <div className="stub-body">not wired yet -- slice pending</div>
    </main>
  );
}
