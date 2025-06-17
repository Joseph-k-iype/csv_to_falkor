/** @type {import('tailwindcss').Config} */
module.exports = {
    content: [
      "./src/**/*.{js,jsx,ts,tsx}",
      "./public/index.html"
    ],
    theme: {
      extend: {
        fontFamily: {
          'sans': ['Inter', 'ui-sans-serif', 'system-ui', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Roboto', 'Helvetica Neue', 'Arial', 'Noto Sans', 'sans-serif'],
        },
        colors: {
          primary: {
            50: '#eff6ff',
            100: '#dbeafe',
            200: '#bfdbfe',
            300: '#93c5fd',
            400: '#60a5fa',
            500: '#3b82f6',
            600: '#2563eb',
            700: '#1d4ed8',
            800: '#1e40af',
            900: '#1e3a8a',
            950: '#172554',
          },
          secondary: {
            50: '#faf5ff',
            100: '#f3e8ff',
            200: '#e9d5ff',
            300: '#d8b4fe',
            400: '#c084fc',
            500: '#a855f7',
            600: '#9333ea',
            700: '#7c3aed',
            800: '#6b21a8',
            900: '#581c87',
            950: '#3b0764',
          },
          glass: {
            light: 'rgba(255, 255, 255, 0.8)',
            medium: 'rgba(255, 255, 255, 0.6)',
            dark: 'rgba(255, 255, 255, 0.4)',
          }
        },
        backdropBlur: {
          xs: '2px',
          '4xl': '72px',
        },
        animation: {
          'bounce': 'bounce 1.4s infinite ease-in-out both',
          'pulse': 'pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite',
          'pulse-soft': 'pulse-soft 2s cubic-bezier(0.4, 0, 0.6, 1) infinite',
          'fade-in': 'fadeIn 0.3s ease-in-out',
          'fade-in-up': 'fadeInUp 0.5s ease-out',
          'fade-in-scale': 'fadeInScale 0.3s ease-out',
          'slide-up': 'slideUp 0.3s ease-out',
          'slide-in-up': 'slideInUp 0.5s ease-out',
          'float': 'float 6s ease-in-out infinite',
          'glow': 'glow 3s ease-in-out infinite',
          'shimmer': 'shimmer 2s infinite',
          'spin-slow': 'spin 3s linear infinite',
          'bounce-sequence': 'bounce-sequence 1.4s infinite ease-in-out both',
        },
        keyframes: {
          fadeIn: {
            '0%': { opacity: '0' },
            '100%': { opacity: '1' },
          },
          fadeInUp: {
            '0%': { opacity: '0', transform: 'translateY(20px)' },
            '100%': { opacity: '1', transform: 'translateY(0)' },
          },
          fadeInScale: {
            '0%': { opacity: '0', transform: 'scale(0.95)' },
            '100%': { opacity: '1', transform: 'scale(1)' },
          },
          slideUp: {
            '0%': { transform: 'translateY(100%)' },
            '100%': { transform: 'translateY(0)' },
          },
          slideInUp: {
            '0%': { opacity: '0', transform: 'translateY(30px)' },
            '100%': { opacity: '1', transform: 'translateY(0)' },
          },
          float: {
            '0%, 100%': { transform: 'translateY(0px)' },
            '50%': { transform: 'translateY(-10px)' },
          },
          glow: {
            '0%, 100%': { boxShadow: '0 0 20px rgba(59, 130, 246, 0.3)' },
            '50%': { boxShadow: '0 0 30px rgba(59, 130, 246, 0.5)' },
          },
          shimmer: {
            '0%': { backgroundPosition: '-200% 0' },
            '100%': { backgroundPosition: '200% 0' },
          },
          'pulse-soft': {
            '0%, 100%': { opacity: '1' },
            '50%': { opacity: '0.8' },
          },
          'bounce-sequence': {
            '0%, 80%, 100%': { 
              transform: 'scale(0) translateY(0)',
              opacity: '0.5'
            },
            '40%': { 
              transform: 'scale(1) translateY(-10px)',
              opacity: '1'
            },
          }
        },
        boxShadow: {
          'glass': '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
          'glass-sm': '0 2px 4px -1px rgba(0, 0, 0, 0.06), 0 1px 2px -1px rgba(0, 0, 0, 0.06)',
          'glass-lg': '0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)',
          'glass-xl': '0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)',
          'glass-2xl': '0 25px 50px -12px rgba(0, 0, 0, 0.25)',
          'inner-light': 'inset 0 1px 0 0 rgba(255, 255, 255, 0.1)',
          'glow': '0 0 20px rgba(59, 130, 246, 0.3)',
          'glow-lg': '0 0 30px rgba(59, 130, 246, 0.4)',
          'soft': '0 2px 8px 0 rgba(31, 38, 135, 0.08)',
          'medium': '0 4px 16px 0 rgba(31, 38, 135, 0.1)',
          'strong': '0 8px 32px 0 rgba(31, 38, 135, 0.15)',
        },
        spacing: {
          '18': '4.5rem',
          '88': '22rem',
          '100': '25rem',
          '104': '26rem',
          '108': '27rem',
          '112': '28rem',
          '116': '29rem',
          '120': '30rem',
        },
        borderRadius: {
          '2xl': '1rem',
          '3xl': '1.5rem',
          '4xl': '2rem',
          '5xl': '2.5rem',
        },
        zIndex: {
          '60': '60',
          '70': '70',
          '80': '80',
          '90': '90',
          '100': '100',
        },
        backgroundImage: {
          'gradient-radial': 'radial-gradient(var(--tw-gradient-stops))',
          'gradient-conic': 'conic-gradient(from 180deg at 50% 50%, var(--tw-gradient-stops))',
          'glass-gradient': 'linear-gradient(135deg, rgba(255, 255, 255, 0.1), rgba(255, 255, 255, 0))',
          'shimmer': 'linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.6), transparent)',
        },
        blur: {
          '4xl': '72px',
          '5xl': '96px',
          '6xl': '128px',
        },
        scale: {
          '102': '1.02',
          '103': '1.03',
          '104': '1.04',
          '105': '1.05',
        },
        screens: {
          'xs': '475px',
          '3xl': '1600px',
        },
        transitionTimingFunction: {
          'smooth': 'cubic-bezier(0.4, 0, 0.2, 1)',
          'bounce-in': 'cubic-bezier(0.68, -0.55, 0.265, 1.55)',
          'ease-out-expo': 'cubic-bezier(0.19, 1, 0.22, 1)',
        },
        transitionDuration: {
          '400': '400ms',
          '600': '600ms',
          '800': '800ms',
          '900': '900ms',
        },
        fontSize: {
          '2xs': ['0.625rem', { lineHeight: '0.75rem' }],
          '3xl': ['1.875rem', { lineHeight: '2.25rem' }],
          '4xl': ['2.25rem', { lineHeight: '2.5rem' }],
          '5xl': ['3rem', { lineHeight: '1' }],
          '6xl': ['3.75rem', { lineHeight: '1' }],
          '7xl': ['4.5rem', { lineHeight: '1' }],
          '8xl': ['6rem', { lineHeight: '1' }],
          '9xl': ['8rem', { lineHeight: '1' }],
        },
        lineHeight: {
          '11': '2.75rem',
          '12': '3rem',
          '13': '3.25rem',
          '14': '3.5rem',
        },
        maxWidth: {
          '8xl': '88rem',
          '9xl': '96rem',
        },
        aspectRatio: {
          '4/3': '4 / 3',
          '3/2': '3 / 2',
          '2/3': '2 / 3',
          '9/16': '9 / 16',
        },
      },
    },
    plugins: [],
  }