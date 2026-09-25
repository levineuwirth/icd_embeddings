// Run with `npm test` (node --test).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { formatRiskPercent, validateAgeInput } from './risk.js';

// Regression: mortality was shown with toFixed(1), so a 0.05% prediction
// read 0.0% against a 0.4% base rate.
test('sub-percent risks keep two significant figures', () => {
  assert.equal(formatRiskPercent(0.0005071241), '0.051%');
  assert.equal(formatRiskPercent(0.0038874548), '0.39%');
  assert.equal(formatRiskPercent(0.000047), '0.0047%');
  assert.equal(formatRiskPercent(0.0000047), '<0.001%');
});

test('risks of 1% and above keep one decimal place', () => {
  assert.equal(formatRiskPercent(0.1068000810), '10.7%');
  assert.equal(formatRiskPercent(0.01), '1.0%');
  assert.equal(formatRiskPercent(1), '100.0%');
});

test('non-numeric risk renders empty', () => {
  assert.equal(formatRiskPercent(undefined), '');
  assert.equal(formatRiskPercent('x'), '');
});

// Regression: ages 0-17 were accepted though the cohort was adults only.
test('minors are rejected with the reason', () => {
  for (const age of ['0', '5', '17']) {
    const r = validateAgeInput(age);
    assert.equal(r.valid, false, age);
    assert.match(r.error, /18 or older/);
  }
  assert.deepEqual(validateAgeInput('18'), { valid: true, error: '', adjustedAge: 18 });
});

test('ages 90-124 cap at 90 and 125+ are rejected', () => {
  assert.equal(validateAgeInput('95').adjustedAge, 90);
  assert.equal(validateAgeInput('124').adjustedAge, 90);
  assert.equal(validateAgeInput('125').valid, false);
});

test('an empty age is not an error', () => {
  assert.deepEqual(validateAgeInput(''), { valid: false, error: '' });
});
