export function getVerdictBadge(reply) {
  if (typeof reply !== 'string') return null;

  const firstLine = reply.trimStart().split(/\r?\n/, 1)[0].trim();
  if (/^The Court Declares:\s*Not\s+Guilty!$/i.test(firstLine)) return 'NTA ✅';
  if (/^The Court Declares:\s*Guilty!$/i.test(firstLine)) return 'YTA 🫵';
  return null;
}
