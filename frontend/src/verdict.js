export function getVerdictBadge(verdict) {
  if (verdict === 'guilty') return 'YTA 🫵';
  if (verdict === 'not_guilty') return 'NTA ✅';
  return null;
}
