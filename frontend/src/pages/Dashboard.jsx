import React from 'react';
import StatsCard from '../components/StatsCard';

export default function Dashboard() {
  const stats = [
    { label: 'Videos Analyzed', value: 0 },
    { label: 'Comments', value: 0 },
    { label: 'Patterns Found', value: 0 },
  ];

  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
      {stats.map(s => <StatsCard key={s.label} {...s} />)}
    </div>
  );
}
