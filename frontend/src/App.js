// src/App.js - Fixed version
import React from 'react';
import { ConnectionProvider } from './contexts/ConnectionContext';
import { SearchProvider } from './contexts/SearchContext';
import { MainLayout } from './components/layout';
import { Header } from './components/layout';
import { Footer } from './components/layout';
import SearchInterface from './components/search/SearchInterface';
import ResultsDisplay from './components/results/ResultsDisplay';
import WelcomeSection from './components/welcome/WelcomeSection';
import { ErrorBoundary } from './components/ui';

/**
 * Main FalkorDB Intelligence Application
 * Refactored for maintainability, performance, and scalability
 */
const FalkorDBApp = () => {
  return (
    <ErrorBoundary>
      <ConnectionProvider>
        <SearchProvider>
          <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-100">
            {/* Background Pattern */}
            <div 
              className="absolute inset-0 opacity-40"
              style={{
                backgroundImage: `url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%239C92AC' fill-opacity='0.03'%3E%3Ccircle cx='30' cy='30' r='1.5'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E")`
              }}
            />
            
            <div className="relative z-10">
              <MainLayout>
                <Header />
                <main className="w-full px-6">
                  <div className="max-w-4xl mx-auto">
                    <SearchInterface />
                    <WelcomeSection />
                    <ResultsDisplay />
                  </div>
                </main>
                <Footer />
              </MainLayout>
            </div>
          </div>
        </SearchProvider>
      </ConnectionProvider>
    </ErrorBoundary>
  );
};

export default FalkorDBApp;