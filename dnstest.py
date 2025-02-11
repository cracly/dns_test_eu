import dns.resolver
import time
import argparse
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from tqdm import tqdm

# List of DNS providers to test
dns_providers = [
    '8.8.8.8',  # Google
    '1.1.1.1',  # Cloudflare
    '1.1.1.2',  # Cloudflare family
    '9.9.9.9',  # quad9
    '149.112.112.112',  # quad9 2
    '9.9.9.11',  # quad9 ecs
    '193.110.81.0',     # dns0.eu 1
    '185.253.5.0',      # dns0.eu 2
    '185.222.222.222',   # dns.sb 1
    '45.11.45.11',   # dns.sb 2
    '91.239.100.100',   # uncensoreddns.org anycast
    '89.233.43.71',  # uncensoreddns.org unicast
]

# List of domains to test
domains = [
    'google.com',
    'apple.com',
    'microsoft.com',
    'netflix.com',
    'youtube.com',
    'google.at',
    'orf.at',
    'tuwien.at',
    'reddit.com',
    'nasa.gov',
    'derstandard.at',
    'bbc.com',
    'diepresse.at',
    'wikipedia.org',
    'twitch.tv',
    'amazon.de',
    'europa.eu',
]


# Number of measurements to take
num_measurements = 50


# Function to test DNS resolution time
def test_dns(dns_server, domain, num_measurements):
    resolver = dns.resolver.Resolver()
    resolver.nameservers = [dns_server]
    times = []

    for _ in tqdm(range(num_measurements), desc=f"Testing {domain} with {dns_server}", leave=False):
        start_time = time.time()
        try:
            answer = resolver.resolve(domain)
            end_time = time.time()
            times.append((end_time - start_time) * 1000)  # Convert to milliseconds
        except Exception as e:
            print(f"Error resolving domain with {dns_server}: {e}")
            return None, None, None

    avg_time = sum(times) / num_measurements
    min_time = min(times)
    max_time = max(times)

    std_dev = np.std(times, ddof=1)  # Sample standard deviation
    std_err = std_dev / np.sqrt(num_measurements)
    confidence_interval = stats.t.interval(0.95, num_measurements - 1, loc=avg_time, scale=std_err)

    return avg_time, min_time, max_time, confidence_interval


# Function to test multiple domains
def test_domains(dns_server, domains, num_measurements, verbose):
    overall_times = []
    domain_stats = []

    for domain in tqdm(domains, desc=f"Testing domains with {dns_server}", leave=False):
        avg_time, min_time, max_time, confidence_interval = test_dns(dns_server, domain, num_measurements)
        if avg_time is not None:
            overall_times.append(avg_time)
            domain_stats.append((domain, avg_time, min_time, max_time, confidence_interval))
            if verbose:
                print(f"DNS Server {dns_server} for domain {domain}:")
                print(f"  avg: {avg_time:.2f} ms, min: {min_time:.2f} ms, max: {max_time:.2f} ms")

    overall_avg = sum(overall_times) / len(overall_times)
    overall_min = min(overall_times)
    overall_max = max(overall_times)
    overall_std_dev = np.std(overall_times, ddof=1)
    overall_std_err = overall_std_dev / np.sqrt(len(overall_times))
    overall_confidence_interval = stats.t.interval(0.95, len(overall_times) - 1, loc=overall_avg, scale=overall_std_err)

    return overall_avg, overall_min, overall_max, overall_confidence_interval, domain_stats


# Function to plot results
def plot_results(dns_providers, overall_stats, domain_stats, verbose):
    # Plot overall stats
    dns_labels = [f"DNS {provider}" for provider in dns_providers]
    avg_times = [stats[0] for stats in overall_stats]
    min_times = [stats[1] for stats in overall_stats]
    max_times = [stats[2] for stats in overall_stats]
    confidence_intervals = [stats[3] for stats in overall_stats]

    plt.figure(figsize=(10, 6))
    plt.bar(dns_labels, avg_times, color='skyblue',
            yerr=[np.array(avg_times) - np.array(min_times), np.array(max_times) - np.array(avg_times)], capsize=5)
    for i, (low, high) in enumerate(confidence_intervals):
        plt.plot([i - 0.2, i + 0.2], [low, low], color='red')
        plt.plot([i - 0.2, i + 0.2], [high, high], color='red')
        plt.plot([i, i], [low, high], color='red')
    plt.xlabel('DNS Providers')
    plt.ylabel('Average Resolution Time (ms)')
    plt.title('Overall Average DNS Resolution Times with 95% Confidence Interval')
    plt.show()

    if verbose:
        # Plot detailed stats for each domain
        for i, provider in enumerate(dns_providers):
            domain_labels = [stats[0] for stats in domain_stats[i]]
            avg_times = [stats[1] for stats in domain_stats[i]]
            confidence_intervals = [stats[4] for stats in domain_stats[i]]

            plt.figure(figsize=(12, 8))
            plt.bar(domain_labels, avg_times, color='lightgreen')
            for j, (low, high) in enumerate(confidence_intervals):
                plt.plot([j - 0.2, j + 0.2], [low, low], color='red')
                plt.plot([j - 0.2, j + 0.2], [high, high], color='red')
                plt.plot([j, j], [low, high], color='red')
            plt.xlabel('Domains')
            plt.ylabel('Average Resolution Time (ms)')
            plt.title(f'Average DNS Resolution Times for {provider} with 95% Confidence Interval')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.show()


# Main function to test all DNS providers
def main(verbose):
    overall_stats = []
    domain_stats = []

    for dns_server in dns_providers:
        overall_avg, overall_min, overall_max, overall_confidence_interval, stats = test_domains(dns_server, domains, num_measurements, verbose)
        overall_stats.append((overall_avg, overall_min, overall_max, overall_confidence_interval))
        domain_stats.append(stats)
        print(f"Overall stats for DNS Server {dns_server}:")
        print(f"  Overall average resolution time: {overall_avg:.2f} ms")
        print(f"  Overall minimum resolution time: {overall_min:.2f} ms")
        print(f"  Overall maximum resolution time: {overall_max:.2f} ms")

    plot_results(dns_providers, overall_stats, domain_stats, verbose)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DNS Benchmark Tool")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()

    main(args.verbose)
