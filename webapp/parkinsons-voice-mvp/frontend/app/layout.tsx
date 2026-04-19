import './globals.css';
import type { Metadata } from 'next';

export const metadata: Metadata = {
  title: "Parkinson's Voice Lessons",
  description: 'Browse self-guided Parkinsons speech therapy lessons and move through them step by step.',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
