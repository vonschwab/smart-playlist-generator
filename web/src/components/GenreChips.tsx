const DEFAULT_CAP = 6;

// Shared chip row for genre lists that arrive pre-ordered (most-specific first)
// from the backend. Shows up to `cap` chips, then a "+N" pill whose title
// tooltip lists the remainder.
export function GenreChips({
  genres,
  chipClass,
  cap = DEFAULT_CAP,
}: {
  genres: string[];
  chipClass: string;
  cap?: number;
}) {
  const shown = genres.slice(0, cap);
  const rest = genres.slice(cap);
  return (
    <>
      {shown.map((g) => (
        <span key={g} className={chipClass}>
          {g}
        </span>
      ))}
      {rest.length > 0 && (
        <span data-testid="genre-overflow" className={chipClass} title={rest.join(", ")}>
          +{rest.length}
        </span>
      )}
    </>
  );
}
