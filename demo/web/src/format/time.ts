function pad(value: number, width = 2): string {
  return String(value).padStart(width, "0");
}

const MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"];

export function hm(date: Date): string {
  return `${pad(date.getHours())}:${pad(date.getMinutes())}`;
}

export function hms(date: Date): string {
  return `${hm(date)}:${pad(date.getSeconds())}.${pad(date.getMilliseconds(), 3)}`;
}

export function dayLabel(date: Date): string {
  return `${MONTHS[date.getMonth()]} ${date.getDate()}`;
}
