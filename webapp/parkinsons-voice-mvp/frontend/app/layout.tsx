import './globals.css';
import type { Metadata } from 'next';

export const metadata: Metadata = {
  title: "Parkinson's Voice MVP",
  description: 'Record and analyze a voice sample for speech practice.',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
