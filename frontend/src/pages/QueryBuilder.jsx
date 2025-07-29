import React, { useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import api from '../utils/api';

export default function QueryBuilder() {
  const [keywords, setKeywords] = useState('');
  const mutation = useMutation({
    mutationFn: (payload) => api.analyze(payload),
  });

  const onSubmit = (e) => {
    e.preventDefault();
    mutation.mutate({ keywords: keywords.split(',').map(k => k.trim()) });
  };

  return (
    <form onSubmit={onSubmit} className="space-y-4">
      <input
        className="border p-2 w-full"
        placeholder="keyword1, keyword2"
        value={keywords}
        onChange={(e) => setKeywords(e.target.value)}
      />
      <button
        type="submit"
        className="px-4 py-2 bg-blue-600 text-white rounded"
        disabled={mutation.isLoading}
      >
        {mutation.isLoading ? 'Analyzing...' : 'Analyze'}
      </button>
      {mutation.isError && (
        <p className="text-red-600">{mutation.error.message}</p>
      )}
      {mutation.data && (
        <pre className="bg-gray-200 p-2 mt-4">
          {JSON.stringify(mutation.data, null, 2)}
        </pre>
      )}
    </form>
  );
}
