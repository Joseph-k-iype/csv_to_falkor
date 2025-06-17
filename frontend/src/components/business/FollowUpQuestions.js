import React from 'react';
import { 
  Lightbulb, 
  Target, 
  BarChart3, 
  Brain, 
  Shield, 
  ChevronRight 
} from 'lucide-react';
import { useFormatters } from '../../hooks/useFormatters';

const FollowUpQuestions = ({ questions, onQuestionSelect, isLoading, metadata }) => {
  const { categorizeQuestions } = useFormatters();
  
  if (!questions || questions.length === 0) return null;
  
  const categorized = categorizeQuestions(questions);
  
  const categoryIcons = {
    strategic: <Target className="w-4 h-4 text-purple-600" />,
    operational: <BarChart3 className="w-4 h-4 text-blue-600" />,
    technical: <Brain className="w-4 h-4 text-green-600" />,
    risk: <Shield className="w-4 h-4 text-red-600" />
  };
  
  return (
    <div className="mb-6">
      <h3 className="text-sm font-medium text-gray-600 mb-3 flex items-center">
        <Lightbulb className="w-4 h-4 mr-2" />
        Intelligent Follow-up Analysis
        {metadata?.total_agents && (
          <span className="ml-2 text-xs bg-blue-100 text-blue-800 px-2 py-1 rounded-full">
            {metadata.total_agents} AI Agents
          </span>
        )}
      </h3>
      
      <div className="grid gap-3">
        {Object.entries(categorized).map(([category, categoryQuestions]) => {
          if (categoryQuestions.length === 0) return null;
          
          return (
            <div key={category}>
              <div className="text-xs font-medium text-gray-500 mb-2 flex items-center">
                {categoryIcons[category]}
                <span className="ml-1 capitalize">{category} Analysis</span>
              </div>
              <div className="space-y-2">
                {categoryQuestions.slice(0, 2).map((question, index) => {
                  const cleanQuestion = question.replace(/^\d+\.\s*/, '').trim();
                  if (!cleanQuestion.includes('?')) return null;
                  
                  return (
                    <button
                      key={`${category}-${index}`}
                      onClick={() => onQuestionSelect(cleanQuestion)}
                      disabled={isLoading}
                      className="group text-left p-3 bg-white/60 hover:bg-white/80 border border-white/30 rounded-xl transition-all duration-200 backdrop-blur-sm hover:shadow-md disabled:opacity-50 disabled:cursor-not-allowed w-full"
                    >
                      <div className="flex items-center justify-between">
                        <span className="text-sm text-gray-700 group-hover:text-gray-900 pr-2">
                          {cleanQuestion}
                        </span>
                        <ChevronRight className="w-4 h-4 text-gray-400 group-hover:text-gray-600 flex-shrink-0" />
                      </div>
                    </button>
                  );
                })}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default React.memo(FollowUpQuestions);