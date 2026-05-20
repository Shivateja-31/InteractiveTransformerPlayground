/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        background: "#09090b",
        surface: "#18181b",
        surfaceHover: "#27272a",
        primary: "#3b82f6",
        primaryHover: "#2563eb",
        accent: "#8b5cf6",
        textPrimary: "#f8fafc",
        textSecondary: "#94a3b8",
      },
      fontFamily: {
        mono: ['"Fira Code"', 'monospace'],
        sans: ['"Inter"', 'sans-serif'],
      }
    },
  },
  plugins: [],
}
