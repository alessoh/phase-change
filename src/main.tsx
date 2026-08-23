/**
 * Entry point for The Ising Lab.
 *
 * Mounts the application shell. Everything else - the Monte Carlo sampler, the
 * classifier, and the four analysis panels - is reached from src/App.tsx.
 */

import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import App from './App';
import './styles.css';

const container = document.getElementById('root');
if (container === null) {
  throw new Error('The Ising Lab could not start: no element with id "root" in the document.');
}

createRoot(container).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
