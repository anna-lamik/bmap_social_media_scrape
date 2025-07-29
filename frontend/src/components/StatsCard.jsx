import React from 'react';
/**
 * Display a single statistic with label and value.
 * @param {{label: string, value: string | number}} props
 */
export default function StatsCard({ label, value }) {
  return (
    <div className="bg-white shadow rounded p-4">
      <p className="text-gray-500">{label}</p>
      <p className="text-xl font-bold" data-testid="value">{value}</p>
    </div>
  );
}
