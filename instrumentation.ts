/** Validate the selected local catalogue when a Node server worker starts. */
export async function register() {
  if (process.env.NEXT_RUNTIME === 'nodejs') {
    const { loadCatalog } = await import('./lib/catalog');
    await loadCatalog();
  }
}
