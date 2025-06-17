import React from 'react';
import { ChevronRight } from 'lucide-react';
import { useSearchContext } from '../../contexts/SearchContext';
import { SAMPLE_QUERIES } from '../../utils/constants';

const WelcomeSection = () => {
  const { hasSearched, setQuery, performAdvancedSearch } = useSearchContext();

  const handleSampleQuery = (sampleQuery) => {
    setQuery(sampleQuery);
    performAdvancedSearch(sampleQuery);
  };

  if (hasSearched) return null;

  return (
    <div className="mt-16 mb-8">
      <h3 className="text-center text-lg font-medium text-gray-700 mb-6">
        Try these advanced analysis questions:
      </h3>
      <div className="grid md:grid-cols-2 gap-4 max-w-3xl mx-auto">
        {SAMPLE_QUERIES.map((sampleQuery, index) => (
          <button
            key={index}
            onClick={() => handleSampleQuery(sampleQuery)}
            className="text-left p-4 bg-white/50 hover:bg-white/70 border border-white/30 rounded-xl transition-all duration-200 backdrop-blur-sm hover:shadow-md group"
          >
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-700 group-hover:text-gray-900">{sampleQuery}</span>
              <ChevronRight className="w-4 h-4 text-gray-400 group-hover:text-gray-600" />
            </div>
          </button>
        ))}
      </div>
    </div>
  );
};

export default WelcomeSection;