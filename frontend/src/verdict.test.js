import { getVerdictBadge } from './verdict';

describe('getVerdictBadge', () => {
  test('maps a guilty verdict to YTA', () => {
    expect(getVerdictBadge('The Court Declares: Guilty!\n\nCourt explanation.')).toBe('YTA 🫵');
  });

  test('maps a not-guilty verdict to NTA', () => {
    expect(getVerdictBadge('The Court Declares: Not Guilty!\n\nCourt explanation.')).toBe('NTA ✅');
  });

  test('accepts harmless leading whitespace and verdict casing', () => {
    expect(getVerdictBadge('  the court declares: not guilty!\r\n\r\nCourt explanation.')).toBe('NTA ✅');
  });

  test.each([
    [''],
    [null],
    ['The Court Declares: Maybe Guilty!'],
    ['Preamble\nThe Court Declares: Guilty!'],
  ])('omits the badge when the verdict is not an unambiguous first line', (reply) => {
    expect(getVerdictBadge(reply)).toBeNull();
  });
});
