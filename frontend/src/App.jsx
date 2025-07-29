import React from 'react';
import { Routes, Route, Link } from 'react-router-dom';
import Dashboard from './pages/Dashboard';
import QueryBuilder from './pages/QueryBuilder';
import ResultsTable from './pages/ResultsTable';
import SentimentChart from './pages/SentimentChart';
import Settings from './pages/Settings';

export default function App() {
  return (
    <div className="min-h-screen flex flex-col">
      <nav className="bg-blue-600 text-white p-4">
        <ul className="flex space-x-4">
          <li><Link to="/">Dashboard</Link></li>
          <li><Link to="/query">Query</Link></li>
          <li><Link to="/results">Results</Link></li>
          <li><Link to="/sentiment">Sentiment</Link></li>
          <li><Link to="/settings">Settings</Link></li>
        </ul>
      </nav>
      <main className="flex-1 p-4">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/query" element={<QueryBuilder />} />
          <Route path="/results" element={<ResultsTable />} />
          <Route path="/sentiment" element={<SentimentChart />} />
          <Route path="/settings" element={<Settings />} />
        </Routes>
      </main>
    </div>
  );
}
