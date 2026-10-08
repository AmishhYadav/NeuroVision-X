/** pushState navigation for the no-router SPA (see main.tsx): updates the URL and fires popstate so Root re-renders. */
export function navigateTo(path: string): void {
  window.history.pushState({}, "", path);
  window.dispatchEvent(new PopStateEvent("popstate"));
}
