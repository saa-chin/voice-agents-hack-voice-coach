import { notFound, redirect } from 'next/navigation';
import { flattenedLessons, getLessonById, getOrderedPhaseKeys } from '../../../data/lessonProgram';

export const dynamicParams = false;

type LessonPageProps = {
  params: Promise<{
    lessonId: string;
  }>;
};

export function generateStaticParams() {
  return flattenedLessons.map((lesson) => ({
    lessonId: lesson.exercise.id,
  }));
}

export default async function LessonPage({ params }: LessonPageProps) {
  const { lessonId } = await params;
  const lesson = getLessonById(lessonId);

  if (!lesson) {
    notFound();
  }

  redirect(`/lessons/${lessonId}/${getOrderedPhaseKeys(lesson)[0]}`);
}
