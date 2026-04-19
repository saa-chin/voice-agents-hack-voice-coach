import { redirect } from 'next/navigation';
import { flattenedLessons, getOrderedPhaseKeys } from '../data/lessonProgram';

export default function HomePage() {
  redirect(`/lessons/${flattenedLessons[0].exercise.id}/${getOrderedPhaseKeys(flattenedLessons[0])[0]}`);
}
