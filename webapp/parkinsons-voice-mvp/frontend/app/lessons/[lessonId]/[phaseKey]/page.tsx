import { notFound } from 'next/navigation';
import LessonPlayer from '../../../../components/LessonPlayer';
import {
  flattenedLessons,
  getLessonById,
  getLessonIndexById,
  getOrderedPhaseKeys,
  type LessonPhaseKey,
} from '../../../../data/lessonProgram';

export const dynamicParams = false;

type LessonPhasePageProps = {
  params: Promise<{
    lessonId: string;
    phaseKey: string;
  }>;
};

export function generateStaticParams() {
  return flattenedLessons.flatMap((lesson) =>
    getOrderedPhaseKeys(lesson).map((phaseKey) => ({
      lessonId: lesson.exercise.id,
      phaseKey,
    })),
  );
}

export default async function LessonPhasePage({ params }: LessonPhasePageProps) {
  const { lessonId, phaseKey } = await params;
  const lesson = getLessonById(lessonId);
  const lessonIndex = getLessonIndexById(lessonId);

  if (!lesson || lessonIndex === -1) {
    notFound();
  }

  const orderedPhaseKeys = getOrderedPhaseKeys(lesson);
  const typedPhaseKey = phaseKey as LessonPhaseKey;

  if (!orderedPhaseKeys.includes(typedPhaseKey) || !lesson.exercise.phases[typedPhaseKey]) {
    notFound();
  }

  return <LessonPlayer lesson={lesson} lessonIndex={lessonIndex} phaseKey={typedPhaseKey} />;
}
