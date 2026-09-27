import { getVerdictBadge } from './verdict';

describe('getVerdictBadge', () => {
  test('maps a guilty verdict to YTA', () => {
    expect(getVerdictBadge('guilty')).toBe('YTA 🫵');
  });

  test('maps a not-guilty verdict to NTA', () => {
    expect(getVerdictBadge('not_guilty')).toBe('NTA ✅');
  });

  test.each([
    [''],
    [null],
    ['GUILTY'],
    ['The Court Declares: Guilty!'],
  ])('omits the badge when the structured verdict is invalid', (verdict) => {
    expect(getVerdictBadge(verdict)).toBeNull();
  });
});
