// Input validation and display formatting for the calculator, kept free of
// React so that `npm test` can check them with node:test.

// The NRD cohort the models were trained on was adults only.
export const MIN_AGE = 18;

// Validate an age as typed. Mirrors backend/main.py's _validate_age:
// under 18 and 125+ are rejected, 90-124 is capped at 90 as in the dataset.
export function validateAgeInput(age) {
  const ageNum = parseInt(age, 10);

  if (age === '' || Number.isNaN(ageNum)) {
    return { valid: false, error: '' };
  }
  if (ageNum < MIN_AGE) {
    return {
      valid: false,
      error: `Age must be ${MIN_AGE} or older: the models were trained on adult discharges.`,
    };
  }
  if (ageNum >= 125) {
    return { valid: false, error: 'Age cannot be 125 or greater.' };
  }
  if (ageNum >= 90) {
    return { valid: true, error: '', adjustedAge: 90 };
  }
  return { valid: true, error: '', adjustedAge: ageNum };
}

// A probability as a percentage. At 1% and above, one decimal place, as
// before; below 1%, two significant figures, because the 30-day mortality
// base rate is about 0.4% and one decimal place showed most predictions as
// 0.0%.
export function formatRiskPercent(p) {
  const pct = Number(p) * 100;
  if (!Number.isFinite(pct)) {
    return '';
  }
  if (pct >= 1) {
    return `${pct.toFixed(1)}%`;
  }
  if (pct < 0.001) {
    return '<0.001%';
  }
  return `${Number(pct.toPrecision(2))}%`;
}
