import { loadCatalog } from '@/lib/catalog';
import Estimator from './estimator';
export const dynamic = 'force-dynamic';
export default async function Page() {
  return <Estimator catalog={await loadCatalog()} />;
}
