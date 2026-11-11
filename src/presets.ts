// Dot colour presets: dim, mid and bright, as hex. tools/phosphor_map.py reads this file too (for its
// PNGs), so keep each on one line, in this form.
export const PRESETS = {
  blue:  ['#24324d', '#6f93e8', '#eef3ff'],
  green: ['#1d3a2a', '#4fe08a', '#e6ffef'],
  amber: ['#3d2810', '#ffb347', '#fff3df'],
  rose:  ['#3d1d2a', '#f07aa6', '#ffeaf2'],
  grey:  ['#3a3a3a', '#d8d8d8', '#ffffff'],
} satisfies Record<string, [string, string, string]>;
